import os
import argparse
import numpy as np
import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification



# ============================================================
# Utility functions
# ============================================================

def classify_tensor(name):
    lname = name.lower()

    if "bias" in lname:
        return "bias"
    elif "norm" in lname or "layernorm" in lname or "ln" in lname:
        return "norm"
    elif "weight" in lname:
        return "weight"
    else:
        return "other"


def convert_bitdepth(q, bitwidth):
    qmin = -(1 << (bitwidth - 1))
    qmax = (1 << (bitwidth - 1)) - 1
    return np.clip(q, qmin, qmax).astype(np.int32)


# ============================================================
# Quantization methods
# ============================================================

def optimal_uniform_quant(x, bitwidth, search_steps=40):
    x = x.astype(np.float32)
    Qmax = (1 << (bitwidth - 1)) - 1

    if x.size == 0 or np.all(x == 0):
        return np.zeros_like(x, dtype=np.int32), 1.0

    std = float(np.std(x))
    if std == 0:
        return np.zeros_like(x, dtype=np.int32), 1.0

    qstep_min = max(std / (1 << (bitwidth + 2)), 1e-12)
    qstep_max = max(std * 4.0, qstep_min * 2.0)

    phi = (1 + np.sqrt(5)) / 2.0
    invphi = 1.0 / phi

    a, b = qstep_min, qstep_max
    c = b - (b - a) * invphi
    d = a + (b - a) * invphi

    def mse(qstep):
        q = np.clip(np.round(x / qstep), -Qmax, Qmax)
        return np.mean((x - q * qstep) ** 2)

    fc, fd = mse(c), mse(d)

    for _ in range(search_steps):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) * invphi
            fc = mse(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) * invphi
            fd = mse(d)

    qstep = (a + b) / 2.0
    q = np.clip(np.round(x / qstep), -Qmax, Qmax)

    return q.astype(np.int32), float(qstep)



def quantize_tensor(arr, use_quant=True, tensor_kind ="weight"):
    numel = arr.size


    if tensor_kind == "buffer":
        return arr, 1.0, 32

    if not use_quant:
        # assume already quantized
        return arr, 1.0, 8


    if numel < 32:
        bitwidth = 12
        qstep = np.max(np.abs(arr)) / (2**(bitwidth - 1) - 1 + 1e-8)
        q = np.round(arr / qstep)

    elif tensor_kind == "weight":
        bitwidth = 8
        q, qstep = optimal_uniform_quant(arr, bitwidth)
    else:
        bitwidth = 12
        q, qstep = optimal_uniform_quant(arr, bitwidth)

    q = convert_bitdepth(q, bitwidth)

    return q.astype(np.int32), qstep, bitwidth


# ============================================================
# Metadata
# ============================================================

def write_metadata(path, tensors):
    with open(path, "w") as f:
        f.write(f"numTensors {len(tensors)}\n\n")

        for t in tensors:
            shape_str = " ".join(map(str, t["shape"]))

            f.write(
                f'{t["id"]} {t["name"]} {t["type"]} '
                f'{t["bitwidth"]} {len(t["shape"])} '
                f'{shape_str} {t["qstep"]}\n'
            )


# ============================================================
# Model loader
# ============================================================

def load_model(name, source="hf", weights=None, quantized=False):
    if source == "torchvision":
        return load_torchvision_model(name, weights, quantized)

    print(f"Loading HuggingFace model: {name}")

    try:
        return AutoModelForCausalLM.from_pretrained(name)
    except:
        pass

    try:
        return AutoModelForSequenceClassification.from_pretrained(name)
    except:
        pass

    try:
        return AutoModel.from_pretrained(name)
    except:
        pass

    raise RuntimeError(f"Could not load model: {name}")

def load_torchvision_model(model_name, weights_name=None, quantized=False):
    import torchvision.models as models

    print(f"Loading torchvision model: {model_name}")

    # Dynamically get constructor
    if not hasattr(models, model_name):
        raise ValueError(f"Unknown torchvision model: {model_name}")

    model_fn = getattr(models, model_name)

    weights = None

    if weights_name is not None:
        # Example: ResNet50_Weights.DEFAULT
        weights = eval(f"models.{weights_name}")

    if quantized:
        model = model_fn(weights=weights, quantize=True)
    else:
        model = model_fn(weights=weights)

    return model
# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--no_quant", action="store_true",
                        help="Skip quantization (already quantized model)")
    
    parser.add_argument("--source", default="hf",
                    choices=["hf", "torchvision"])

    parser.add_argument("--weights", default=None,
                        help="Torchvision weights enum (e.g., ResNet50_Weights.DEFAULT)")

    parser.add_argument("--quantized", action="store_true",
                        help="Load quantized torchvision model")

    args = parser.parse_args()

    model = load_model(
        args.model,
        source=args.source,
        weights=args.weights,
        quantized=args.quantized
    )
    model.eval()

    bin_dir = os.path.join(args.out_dir, "binaries")
    meta_file = os.path.join(args.out_dir, "tensor.meta")

    os.makedirs(bin_dir, exist_ok=True)

    tensor_meta_list = []
    tensor_id = 0

    # --- PARAMETERS ---
    for name, param in tqdm(model.named_parameters(), desc="parameters"):
        tensor_kind = "weight" if "weight" in name.lower() else "bias" if "bias" in name.lower() else "other"

        if (args.quantized and hasattr(param, "int_repr")):
            arr = param.int_repr().cpu().numpy().astype(np.int32)
            q, qstep, bitwidth = quantize_tensor(
                arr,
                use_quant=False, # no quantization for already quantized models
                tensor_kind=tensor_kind
            )
        else:
            arr = param.detach().cpu().numpy().astype(np.float32)
            q, qstep, bitwidth = quantize_tensor(
                arr,
                use_quant=True,
                tensor_kind=tensor_kind
            )

    
        

        tensor_meta_list.append({
            "id": tensor_id,
            "name": name,
            "type": tensor_kind,
            "bitwidth": bitwidth,
            "shape": list(q.shape),
            "qstep": qstep
        })

        # Save binary
        tensor_file = f"{name}.bin"
        np.ascontiguousarray(q.astype(np.int32)).tofile(os.path.join(bin_dir, tensor_file))

        tensor_id += 1

    # --- BUFFERS ---
    for name, buf in tqdm(model.named_buffers(), desc="buffers"):
        arr = buf.detach().cpu().numpy().astype(np.float32)
        tensor_kind = "buffer"

        # Always cast to int32, never quantize
        q, qstep, bitwidth = quantize_tensor(
            arr,
            use_quant=False,
            tensor_kind=tensor_kind
        )

        tensor_meta_list.append({
            "id": tensor_id,
            "name": name,
            "type": tensor_kind,
            "bitwidth": bitwidth,
            "shape": list(q.shape),
            "qstep": qstep
        })

        # Save binary
        tensor_file = f"{name}.bin"
        np.ascontiguousarray(q.astype(np.int32)).tofile(os.path.join(bin_dir, tensor_file))

        tensor_id += 1

    write_metadata(meta_file, tensor_meta_list)

    print("\nDone.")
    print("Binaries:", bin_dir)
    print("Metadata:", meta_file)
    print("Total tensors:", len(tensor_meta_list))


if __name__ == "__main__":
    main()