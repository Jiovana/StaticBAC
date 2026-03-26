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


# Path to your CSV file
qstep_file = r"C:\Users\gomes\OneDrive\Documentos\GitHub\nncodec2_work\example\compression scripts\bert_quant_eval_mixed_run5\compression_results.csv"

def build_id_to_name_map(model, decoded_meta):
    sd = model.state_dict()
    keys = list(sd.keys())

    if len(keys) != len(decoded_meta):
        raise RuntimeError(
            f"Mismatch: model has {len(keys)} tensors but decoded_meta has {len(decoded_meta)}"
        )

    id_to_name = {}

    for i, t in enumerate(decoded_meta):
        tensor_id = t["idx"]
        name = keys[i]

        # 🔒 Strong safety check
        if list(sd[name].shape) != list(t["shape"]):
            raise RuntimeError(
                f"Shape mismatch at ID {tensor_id}: "
                f"{name} {list(sd[name].shape)} vs decoded {list(t['shape'])}"
            )

        id_to_name[tensor_id] = name

    print("ID → name mapping built and verified.")
    return id_to_name

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
def load_by_id(model, decoded_meta, folder, qstep_file):
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

    sd = model.state_dict()

    id_to_name = build_id_to_name_map(model, decoded_meta)

    with torch.no_grad():
        for t in decoded_meta:
            tensor_id = t["idx"]
            param_name = id_to_name[tensor_id]
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
                raise RuntimeError(f"Shape mismatch for {param_name}")

            # Copy reconstructed tensor to model
            sd[param_name].copy_(tensor_torch)

            if not torch.allclose(sd[param_name], tensor_torch):
                raise RuntimeError(f"Copy failed for {param_name}")

    print("All tensors loaded and reconstructed successfully.")





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
    META_PATH = os.path.join(DECODED_FOLDER, "decoded_tensors.meta")

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    print("Reading decoded metadata...")
    decoded_meta = read_decoded_meta(META_PATH)
    print("Decoded tensors:", len(decoded_meta))

    for name, param in model.named_parameters():
        print(name, torch.mean(param).item())
        break

    print("Reconstructing weights...")
    load_by_id(model, decoded_meta, DECODED_FOLDER, qstep_file)

    for name, param in model.named_parameters():
        print(name, torch.mean(param).item())
        break


    model_orig = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    for (n1, p1), (n2, p2) in zip(model_orig.named_parameters(), model.named_parameters()):
        diff = torch.max(torch.abs(p1 - p2)).item()
        print(n1, diff)
        break


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_dataset("sst2", split="validation[:1000]")

    acc = evaluate(model, tokenizer, dataset)

    print("\nAccuracy:", acc)