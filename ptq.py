from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
import torch
import modelopt.torch.quantization as mtq
from tqdm import tqdm

# --------------------------
# Define the calibration dataset loader
# --------------------------

def get_dataloader(num_samples=512, batch_size=8, seed=42):
    """
    Create a calibration dataloader from a mixture of:
      - abisee/cnn_dailymail
      - nvidia/Nemotron-Post-Training-Dataset-v2

    Parameters:
        num_samples (int): total number of calibration samples.
        batch_size (int): number of samples per batch.
        seed (int): random seed for reproducibility.
    """
    # 1. Load a subset of both datasets
    cnn_ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train[:5%]")
    nemo_ds = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v2", split="train[:5%]")

    # 2. Normalize fields to a common format (we’ll use “text”)
    def normalize_cnn(ex):
        return {"text": ex["article"]}

    def normalize_nemo(ex):
        # Each example usually has 'prompt' and 'response' fields
        # Concatenate for a realistic text distribution
        return {"text": (ex.get("prompt", "") + " " + ex.get("response", "")).strip()}

    cnn_ds = cnn_ds.map(normalize_cnn, remove_columns=cnn_ds.column_names)
    nemo_ds = nemo_ds.map(normalize_nemo, remove_columns=nemo_ds.column_names)

    # 3. Concatenate and shuffle
    mixed_ds = concatenate_datasets([cnn_ds, nemo_ds]).shuffle(seed=seed)

    # 4. Select only num_samples samples for calibration
    mixed_ds = mixed_ds.select(range(min(num_samples, len(mixed_ds))))

    # 5. Collate into tensors — model likely expects tokenized inputs
    # For simplicity, just return raw text batches here;
    # the model’s forward may include tokenization internally.
    def collate_fn(batch):
        return [item["text"] for item in batch]

    dataloader = DataLoader(
        mixed_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return dataloader

def get_cnn_dataloader(model_name, num_samples=512, batch_size=8, seed=42, max_length=512):
    """
    Create a calibration dataloader using only the CNN/DailyMail dataset.
    Tokenizes the articles for use with transformer models.
    """
    # 1. Load the dataset
    cnn_ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train[:5%]")

    # 2. Normalize to a single text field
    def normalize_cnn(ex):
        return {"text": ex["article"]}

    cnn_ds = cnn_ds.map(normalize_cnn, remove_columns=cnn_ds.column_names)

    # 3. Shuffle and select a subset
    cnn_ds = cnn_ds.shuffle(seed=seed).select(range(min(num_samples, len(cnn_ds))))

    # 4. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 5. Tokenize samples
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized_ds = cnn_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # 6. Set PyTorch format
    tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # 7. Build DataLoader
    dataloader = DataLoader(
        tokenized_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    return dataloader

# --------------------------
# Define the forward loop
# --------------------------

# Setup the model and calibration set
model_name = "Qwen/Qwen3-8B"
device = torch.device("cuda")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
calib_set = get_cnn_dataloader(model_name)


def forward_loop(model):
    """
    Runs the model forward over the calibration set (no gradients).
    """
    model.eval()
    for batch in tqdm(calib_set):
        # If model expects tokenized text, handle here:
        # inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        # model(**inputs)
        batch_cuda = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        model(**batch_cuda)

USE_SEARCH = True
os.environ["USE_SCALE_SEARCH"] = "1" if USE_SEARCH else "0"

# Verify environment variable is set
print(f"Environment variable USE_SCALE_SEARCH is set to: {os.getenv('USE_SCALE_SEARCH')}")

# Run your code here
print("Starting quantization...")
model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop=forward_loop)
print("Quantization complete!")

# Verify quantization by checking if quantizers are present
has_quantizers = any("quantizer" in name for name, _ in model.named_modules())
print(f"Model has quantizers: {has_quantizers}")

# PTQ with in-place replacement to quantized modules
model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop)

from modelopt.torch.export import export_hf_checkpoint
export_dir = f"./llama-quant-{"search" if USE_SEARCH else "nosearch"}-aligned/"
print(f"Exporting quantized model to {export_dir}...")

with torch.inference_mode():
    export_hf_checkpoint(
        model,  # The quantized model.
        export_dir=export_dir
    )

# Save tokenizer to the export directory
print("Saving tokenizer...")
tokenizer.save_pretrained(export_dir)

print("Export complete!")

