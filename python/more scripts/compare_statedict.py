import torch
import numpy as np
import os
import csv

from torchvision import models

# reuse your helpers
# read_decoded_meta, load_tensor, load_qsteps, BITWIDTH_MAP
BITWIDTH_MAP = {
    0:4, 1:8, 2:12, 3:16, 4:20, 5:24, 6:32
}


def recommend_batch_size(model_name, device="cpu"):
    if device == "cpu":
        if "resnet" in model_name:
            return 128
        if "efficientnet" in model_name:
            return 256
        if "vit" in model_name:
            return 128
    return 64


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
def load_tensor(path, shape):

    arr = np.fromfile(path, dtype=np.int32)

    expected = np.prod(shape)

    if arr.size != expected:
        raise RuntimeError(
            f"{path}: expected {expected} values but found {arr.size}"
        )

    arr = arr.reshape(shape)

    return torch.from_numpy(arr)


# ------------------------------------------------------------
# load qsteps
# ------------------------------------------------------------
def load_qsteps(csv_file, target_bits=8):

    tensor_qsteps = {}

    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)

        for row in reader:

            name = row['param_name'].strip()
            bits = int(row['bits_quant'].strip())
            qstep = float(row['qstep'])

            if bits != target_bits:
                continue

            tensor_qsteps[name] = qstep

    print(f"Loaded {len(tensor_qsteps)} qsteps for {target_bits}-bit")

    return tensor_qsteps



def build_reconstructed_state_dict(model,
                                   decoded_meta_param,
                                   decoded_meta_buffer,
                                   folder_param,
                                   folder_buffer,
                                   qstep_param_file,
                                   qstep_buffer_file,
                                   bitwidth=8):

    sd_recon = {}

    # ---- params ----
    qsteps_param = load_qsteps(qstep_param_file, target_bits=bitwidth)
    params = list(model.named_parameters())

    filtered_meta = [t for t in decoded_meta_param if t["bitwidth"] == bitwidth]

    print("\n--- PARAM ORDER CHECK ---")
    for i, t in enumerate(filtered_meta):

        name, ref_tensor = params[i]

        bin_path = os.path.join(folder_param, t["filename"])
        tensor = load_tensor(bin_path, t["shape"])

        if name not in qsteps_param:
            raise ValueError(f"Missing qstep for {name}")
        q = qsteps_param[name]
        print(name, q)
        tensor = tensor.float() * q
        tensor = tensor.to(ref_tensor.dtype)

        sd_recon[name] = tensor

        print(f"{i:03d} | {name} | meta_shape={t['shape']} | model_shape={tuple(ref_tensor.shape)}")

    # ---- buffers ----
    qsteps_buf = load_qsteps(qstep_buffer_file, target_bits=bitwidth)
    buffers = list(model.named_buffers())

    print("\n--- BUFFER ORDER CHECK ---")
    for i, t in enumerate(decoded_meta_buffer):

        name, ref_tensor = buffers[i]

        bin_path = os.path.join(folder_buffer, t["filename"])
        tensor = load_tensor(bin_path, t["shape"])

        
        if t["bitwidth"] != 32:
            if name not in qsteps_buf:
                raise ValueError(f"Missing qstep for {name}")
            q = qsteps_buf[name]
            print(name, q)

            tensor = tensor.float() * q
        else:
            tensor = tensor.float()

        tensor = tensor.to(ref_tensor.dtype)

        sd_recon[name] = tensor

        print(f"{i:03d} | {name} | meta_shape={t['shape']} | model_shape={tuple(ref_tensor.shape)}")

    return sd_recon


def compare_state_dicts(model, sd_recon):

    sd_orig = model.state_dict()

    print("\n==============================")
    print("STATE DICT COMPARISON")
    print("==============================\n")

    mismatches = []

    for i, (name, orig) in enumerate(sd_orig.items()):

        if name not in sd_recon:
            print(f"[MISSING] {name}")
            continue

        recon = sd_recon[name]

        if orig.shape != recon.shape:
            print(f"[SHAPE MISMATCH] {name}: {orig.shape} vs {recon.shape}")
            continue

        diff = (orig - recon).abs().max().item()

        print(f"{i:03d} | {name:40s} | max diff = {diff:.3e}")

        if diff > 1e-4:
            mismatches.append(name)

    print("\nLarge mismatches:", len(mismatches))


def detect_permutation(model, sd_recon, top_k=3):
    """
    Try to detect if a tensor matches a DIFFERENT tensor (order bug)
    """

    print("\n==============================")
    print("PERMUTATION CHECK")
    print("==============================\n")

    sd_orig = model.state_dict()
    names = list(sd_orig.keys())

    for name in names[:20]:  # limit to first 20 for sanity

        recon = sd_recon[name]

        best_match = None
        best_diff = 1e9

        for other_name, orig in sd_orig.items():

            if orig.shape != recon.shape:
                continue

            diff = (orig - recon).abs().max().item()

            if diff < best_diff:
                best_diff = diff
                best_match = other_name

        print(f"{name}")
        print(f"  best match: {best_match}")
        print(f"  diff: {best_diff:.3e}")

        if best_match != name:
            print("  ⚠️ ORDER MISMATCH DETECTED\n")
        else:
            print("  OK\n")


def main():

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()

    # paths
    folder_param = "resnet_tensors_decoded"
    folder_buffer = "resnet_buffers_decoded"

    meta_param = read_decoded_meta(os.path.join(folder_param, "decoded_tensors.meta"))
    meta_buffer = read_decoded_meta(os.path.join(folder_buffer, "decoded_tensors.meta"))

    qstep_param = r"C:\Users\gomes\OneDrive\Documentos\GitHub\nncodec2_work\example\compression scripts\multi_model_quant_eval_run5\resnet50_compression.csv"
    qstep_buffer = r"C:\Users\gomes\OneDrive\Documentos\GitHub\nncodec2_work\example\compression scripts\multi_model_quant_eval_run5\resnet50_buffers_compression.csv"


    # build reconstructed dict
    sd_recon = build_reconstructed_state_dict(
        model,
        meta_param,
        meta_buffer,
        folder_param,
        folder_buffer,
        qstep_param,
        qstep_buffer
    )

    # compare
    compare_state_dicts(model, sd_recon)

    # detect permutation
    detect_permutation(model, sd_recon)


if __name__ == "__main__":
    main()