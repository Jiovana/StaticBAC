import csv
import os
import torch
import math
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import transformers
transformers.logging.set_verbosity_error()


BITWIDTH_MAP = {
    0:4, 1:8, 2:12, 3:16, 4:20, 5:24, 6:32
}


# Path to your CSV file
qstep_file = r"C:\Users\gomes\OneDrive\Documentos\GitHub\nncodec2_work\example\compression scripts\gpt_quant_eval_mixed_run5\compression_results.csv"


# ------------------------------------------------------------
# read decoded meta
# ------------------------------------------------------------
def read_decoded_meta(path):

    tensors = []

    with open(path) as f:
        lines = f.readlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if line.startswith("numTensors"):
            continue

        parts = line.split()

        # safety check
        if len(parts) < 6:
            continue

        idx = int(parts[0])
        filename = parts[1]
        bw_enum = int(parts[3])
        dims = int(parts[4])

        shape = tuple(map(int, parts[5:5+dims]))

        tensors.append({
            "idx": idx,
            "filename": filename,
            "bitwidth": BITWIDTH_MAP[bw_enum],
            "shape": shape
        })

    return tensors


# ------------------------------------------------------------
# load tensor binary
# ------------------------------------------------------------
def load_tensor(path, bitwidth, shape):

    # decoded tensors appear to be stored as int32
    dtype = np.int32

    arr = np.fromfile(path, dtype=dtype)

    expected = np.prod(shape)

    if arr.size != expected:
        raise RuntimeError(
            f"{path}: expected {expected} values but found {arr.size}"
        )

    arr = arr.reshape(shape)

    return torch.from_numpy(arr)


# ------------------------------------------------------------
# inject tensors by order
# ------------------------------------------------------------
def load_by_order(model, decoded_meta, folder, qstep_file):
    """
    Load decoded tensors into a model by order, reconstructing them using qstep.
    
    Args:
        model: PyTorch model.
        decoded_meta: List of dictionaries with keys ['filename', 'bitwidth', 'shape'] for each tensor.
        folder: Folder where decoded tensors are stored.
        qstep_file: CSV file containing qsteps for all tensors.
    """
    # Load qsteps into a dict: param_name -> qstep
    tensor_qsteps = {}
    with open(qstep_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['param_name']
            qstep = float(row['qstep'])
            tensor_qsteps[name] = qstep

    params = dict(model.named_parameters())
    keys = list(params.keys())
    assert len(keys) == len(decoded_meta), f"State dict has {len(keys)} keys, but decoded_meta has {len(decoded_meta)}"

    with torch.no_grad():
        for i, t in enumerate(decoded_meta):
            param_name = keys[i]
            bin_path = os.path.join(folder, t["filename"])

            # Load integer tensor
            tensor = load_tensor(bin_path, t["bitwidth"], t["shape"])

            # Reconstruct using qstep
            if param_name not in tensor_qsteps:
                raise ValueError(f"qstep for {param_name} not found in qstep file")
            qstep = tensor_qsteps[param_name]

            tensor = tensor.to(torch.float32) * qstep

            # Ensure dtype matches model
            tensor_torch = tensor.to(params[param_name].dtype)

            if params[param_name].shape != tensor_torch.shape:
                print("Shape mismatch:", param_name, params[param_name].shape, tensor_torch.shape)
                continue

            # Copy reconstructed tensor to model
            params[param_name].data.copy_(tensor_torch)

    print("All tensors loaded and reconstructed successfully.")

# ------------------------------------------------------------
# inject tensors by ID (safe for extra final head)
# ------------------------------------------------------------
def load_by_id(model, decoded_meta, folder, qstep_file):
    """
    Load decoded tensors into a model by param name, reconstructing them using qstep.
    Handles cases where the model has extra parameters not present in the decoded tensors.

    Args:
        model: PyTorch model.
        decoded_meta: List of dicts with keys ['filename', 'bitwidth', 'shape'].
        folder: Folder where decoded tensors are stored.
        qstep_file: CSV file containing qsteps for all tensors.
    """
    # Load qsteps into a dict: param_name -> qstep
    tensor_qsteps = {}
    with open(qstep_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['param_name']
            qstep = float(row['qstep'])
            tensor_qsteps[name] = qstep

    # Build ID -> param_name mapping
    sd = model.state_dict()
    id_to_name = {i: name for i, name in enumerate(sd.keys())}
    print("ID → name mapping built.")

    # Only iterate over decoded tensors
    with torch.no_grad():
        for i, t in enumerate(decoded_meta):
            # Defensive: skip if decoded tensors > model parameters
            if i >= len(id_to_name):
                print(f"Skipping extra decoded tensor {t['filename']} (no matching model param)")
                continue

            param_name = id_to_name[i]
            bin_path = os.path.join(folder, t["filename"])

            # Load integer tensor
            tensor = load_tensor(bin_path, t["bitwidth"], t["shape"])

            # Reconstruct using qstep
            if param_name not in tensor_qsteps:
                raise ValueError(f"qstep for {param_name} not found in qstep file")
            qstep = tensor_qsteps[param_name]

            tensor = tensor.to(torch.float32) * qstep

            # Ensure dtype matches model
            tensor_torch = tensor.to(sd[param_name].dtype)

            if sd[param_name].shape != tensor_torch.shape:
                print("Shape mismatch:", param_name, sd[param_name].shape, tensor_torch.shape)
                continue

            # Copy reconstructed tensor to model
            sd[param_name].copy_(tensor_torch)

    print("All decoded tensors loaded and reconstructed successfully.")
    print("Any extra model parameters (e.g., final head) were left unchanged.")



def load_qsteps(csv_file):
    # Create a dictionary to store qstep per tensor
    tensor_qsteps = {}

    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract tensor name and qstep
            name = row['param_name']
            qstep = float(row['qstep'])
            shape_str = row['shape']  # e.g., "(768, 3072)"
            shape = tuple(int(x) for x in shape_str.strip("()").split(","))

            # Store qstep and shape in dict
            tensor_qsteps[name] = {
                "qstep": qstep,
                "shape": shape,
                "dtype": np.float32  # you can adjust if needed
            }
    return tensor_qsteps

def perplexity_sliding_window(model, tokenizer, text, max_length=512, stride=512):
    # Tokenize once (no truncation)
    enc = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=True)
    input_ids = enc["input_ids"][0]  # shape: (n_tokens,)
    n_tokens = input_ids.size(0)

    model.eval()
    total_loss = 0.0
    total_count = 0

    # stride should be <= max_length; for full overlap use stride < max_length
    for begin_idx in tqdm(range(0, n_tokens, stride), desc="Chunks"):
        end_idx = min(begin_idx + max_length, n_tokens)
        input_ids_chunk = input_ids[begin_idx:end_idx]

        # prepare labels: mask context tokens with -100
        # we set labels equal to input ids, but tokens before the "target" region are -100.
        labels = input_ids_chunk.clone()
        # If begin_idx == 0, there is no "context" to mask. If we want some context-only region,

        # A more robust approach (from HF) is:
        chunk_len = input_ids_chunk.size(0)


        # Put into model (batch dimension)
        input_batch = input_ids_chunk.unsqueeze(0).to(model.device)
        labels = labels.unsqueeze(0).to(model.device)

        with torch.no_grad():
            outputs = model(input_batch, labels=labels)
            # outputs.loss is the average loss over non -100 labels in this batch
            # Multiply by number of predicted tokens in this chunk to get summed NLL
            # We must count valid labels (labels != -100)
            valid = (labels != -100).sum().item()
            if valid == 0:
                continue
            nll = outputs.loss.item() * valid
            total_loss += nll
            total_count += valid

        if end_idx == n_tokens:
            break

    avg_nll = total_loss / total_count
    ppl = math.exp(avg_nll)
    return ppl, avg_nll, total_count

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":

    print("Loading model...")
    model_name = "openai-community/openai-gpt"   # or your GPT-1 checkpoint
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    DECODED_FOLDER = "gpt_decoded"
    META_PATH = os.path.join(DECODED_FOLDER, "GPT_decoded_tensors.meta")


    print("Reading decoded metadata...")
    decoded_meta = read_decoded_meta(META_PATH)
    print("Decoded tensors:", len(decoded_meta))

    print("Reconstructing weights...")
    load_by_id(model, decoded_meta, DECODED_FOLDER, qstep_file)

    model_orig = AutoModelForCausalLM.from_pretrained(model_name)

    for (n1, p1), (n2, p2) in zip(model_orig.named_parameters(), model.named_parameters()):
        diff = torch.max(torch.abs(p1 - p2)).item()
        print(n1, diff)  # diff > 0 → tensor replaced


    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use full validation split for robust estimate
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n".join([t for t in dataset["text"] if t.strip()])

    ppl, avg_nll, token_count = perplexity_sliding_window(model, tokenizer, text, max_length=512, stride=256)
    print(f"Perplexity: {ppl:.4f}, avg_nll: {avg_nll:.6f}, tokens: {token_count}")