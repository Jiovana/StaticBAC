import csv
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset


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


# ------------------------------------------------------------
# evaluation
# ------------------------------------------------------------
def evaluate(model, tokenizer, dataset):

    model.eval()

    correct = 0
    total = 0

    for item in tqdm(dataset):

        inputs = tokenizer(
            item["sentence"],
            return_tensors="pt",
            truncation=True
        )

        with torch.no_grad():
            logits = model(**inputs).logits

        pred = torch.argmax(logits, dim=-1).item()

        correct += pred == item["label"]
        total += 1

    return correct / total


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":

    MODEL_NAME = "textattack/bert-base-uncased-SST-2"
    DECODED_FOLDER = "bert_decoded"


    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to("cpu").eval()

    npz = np.load("bert_reconstructed.npz")
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

    model.load_state_dict(sd, strict=True)
    model.eval()
   
    print(f"Model BERT successfully reconstructed for {bits} bits !")


    # === Inference ===

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_dataset("sst2", split="validation[:1000]")

    acc = evaluate(model, tokenizer, dataset)

    print("\nAccuracy:", acc)