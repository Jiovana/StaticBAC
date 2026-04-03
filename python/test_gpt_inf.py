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

def check_reconstruction_errors(model, loaded_tensors):
    """
    Compare reconstructed tensors (loaded_tensors) against the original model.

    Args:
        model: Original pre-trained PyTorch model (state_dict is used for reference)
        loaded_tensors: dict mapping param names -> numpy arrays (reconstructed)

    Returns:
        errors: dict mapping param name -> dict(max_abs, mean_abs)
    """
    errors = {}
    sd = model.state_dict()
    print(f"Names in sd: {sd.keys()}")
    for name, recon in loaded_tensors.items():
        if name not in sd:
            print(f"WARNING: {name} not found in model state_dict, skipping.")
            continue
        orig = sd[name].cpu().numpy().astype(np.float32)
        if orig.shape != recon.shape:
            print(f"WARNING: shape mismatch for {name}: model {orig.shape}, reconstructed {recon.shape}")
            continue
        abs_diff = np.abs(orig - recon)
        max_err = np.max(abs_diff)
        mean_err = np.mean(abs_diff)
        errors[name] = {"max_abs": max_err, "mean_abs": mean_err}

       # print(f"original: {orig} | recon: {recon} \n")

    return errors

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
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    DECODED_FOLDER = "gpt_decoded"

    npz = np.load("gpt_reconstructed.npz")
    bits = 8

    loaded = {}
    for k in npz.files:
        raw_name = k.split("_", 2)[-1]
        if raw_name.startswith("param_") or raw_name.startswith("buffer_"):
            print(f"WARNING: key {k} still has param_/buffer_ prefix, skipping.")
            continue
        arr = npz[k].astype(np.float32)
        loaded[raw_name] = arr

    print("Loaded tensors:", len(loaded))
    print("Example keys:", list(loaded.keys())[:10])    

    sd = model.state_dict()

    # Check reconstruction errors
    errors = check_reconstruction_errors(model, loaded)

    print("=== Reconstruction error summary ===")
    max_errors = []
    mean_errors = []
    for name, e in errors.items():
        print(f"{name}: max={e['max_abs']:.6f}, mean={e['mean_abs']:.6f}")
        max_errors.append(e['max_abs'])
        mean_errors.append(e['mean_abs'])

    print(f"Overall max error: {np.max(max_errors):.6f}")
    print(f"Overall mean error: {np.mean(mean_errors):.6f}")

    # overwrite matching keys
    for name, val in loaded.items():
        if name in sd:
            sd[name] = torch.from_numpy(val).to(dtype=sd[name].dtype)
        else:
            print("WARNING: stored key not in model:", name)

    model.load_state_dict(sd, strict=False)
    model.eval()
   
    print(f"Model GPT successfully reconstructed for {bits} bits !")

    # === Inference ===

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use full validation split for robust estimate
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n".join([t for t in dataset["text"] if t.strip()])

    ppl, avg_nll, token_count = perplexity_sliding_window(model, tokenizer, text, max_length=512, stride=256)
    print(f"Perplexity: {ppl:.4f}, avg_nll: {avg_nll:.6f}, tokens: {token_count}")