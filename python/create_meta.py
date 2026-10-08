"""
===============================================================================
Neural Network Tensor Export & Quantization Tool
===============================================================================

Description:
------------
This script extracts tensors from PyTorch, Torchvision, or Hugging Face
models and exports:

1. Binary tensor files (.bin), stored as int32
2. A metadata file describing all exported tensors

The output is designed to be consumed by the StaticBAC encoder/decoder
pipeline.

The script supports uniform 8-bit quantization of model parameters using
either a reconstruction-error (MSE) objective or a rate-distortion (RD)
objective. Non-parameter buffers are not quantized and are exported with
32-bit representation.

-------------------------------------------------------------------------------
Supported Model Sources:
-------------------------------------------------------------------------------

1. Hugging Face (default)
   - NLP / transformer models
   - Examples:
        bert-base-uncased
        gpt2
        meta-llama/Llama-2-7b-hf

2. Torchvision
   - Image models
   - Examples:
        resnet50
        mobilenet_v2
        vgg19
        efficientnet_b0

-------------------------------------------------------------------------------
Usage:
-------------------------------------------------------------------------------

    python create_meta.py \
        --model <model_name_or_path> \
        --out_dir <output_directory> \
        [--source hf|torchvision] \
        [--weights <weights_enum>] \
        [--quantized] \
        [--no_quant] \
        [--quantizer mse|rd] \
        [--lambda_rd <value>]

-------------------------------------------------------------------------------
Arguments:
-------------------------------------------------------------------------------

--model <string>   (required)
    Model name or path.

--out_dir <path>   (required)
    Output directory where the binary tensors and metadata file are stored.

--source <string>  (default: hf)
    Model source:
        hf           → Hugging Face models
        torchvision  → Torchvision models

--weights <string> (optional)
    Torchvision weights enum.

    Example:
        ResNet50_Weights.DEFAULT

--quantized (flag)
    Load a pre-quantized Torchvision model and extract its integer
    representation using int_repr().

--no_quant (flag)
    Skip the quantization procedure and assume the parameter tensors are
    already quantized. Parameters processed through this path are assigned
    bitwidth=8 and qstep=1.0.

--quantizer <string> (default: mse)
    Select the parameter quantization objective:

        mse → minimize normalized mean-squared error
        rd  → minimize normalized MSE + lambda_rd × normalized entropy

--lambda_rd <value> (default: 0.15)
    Weight of the entropy term in the RD quantization objective.

    lambda_rd = 0 corresponds to pure MSE optimization.
    Increasing lambda_rd places greater emphasis on reducing the entropy
    of the quantized representation and can therefore increase
    reconstruction distortion.

-------------------------------------------------------------------------------
Outputs:
-------------------------------------------------------------------------------

<out_dir>/
├── binaries/
│   ├── layer1.weight.bin
│   ├── layer1.bias.bin
│   └── ...
└── tensor.meta

All binary tensor files are stored as int32 values, regardless of the
logical bitwidth specified in tensor.meta.

-------------------------------------------------------------------------------
Metadata Format:
-------------------------------------------------------------------------------

    numTensors <N>

    <id> <name> <type> <bitwidth> <numDims> <shape...> <qstep>

Example:
    0 conv1.weight weight 8 4 64 3 7 7 0.02

The metadata records the logical bitwidth used by the StaticBAC coder and
the quantization step required for reconstruction.

-------------------------------------------------------------------------------
Tensor Handling:
-------------------------------------------------------------------------------

1. Parameters (named_parameters):
    - Parameters are quantized to 8-bit signed integers by default.
    - The quantization objective is selected with --quantizer.
    - MSE quantization minimizes normalized reconstruction error.
    - RD quantization minimizes normalized MSE plus an entropy term.
    - Quantized integer values are stored as int32 in the output files.

2. Buffers (named_buffers):
    - Buffers are not quantized.
    - They are exported with bitwidth=32 and qstep=1.0.
    - The current implementation converts the buffer values to float32 and
      subsequently stores them as int32.

3. Pre-quantized Torchvision models:
    - When --quantized is specified, integer tensor values are extracted
      using int_repr().
    - No additional quantization is applied to those values.

-------------------------------------------------------------------------------
Quantization:
-------------------------------------------------------------------------------

For parameters processed by the standard quantization path:

- Symmetric uniform quantization is used.
- The quantization bitwidth is 8 bits.
- The quantization step is searched using golden-section search.
- MSE mode minimizes normalized mean-squared error.
- RD mode minimizes:

      J = D + lambda_rd * H

  where D is normalized MSE and H is the empirical entropy of the
  quantized representation, normalized by the quantization bitwidth.

The final quantized values are stored as int32 for compatibility with the
StaticBAC C++ pipeline. The logical bitwidth recorded in the metadata is
used by the codec when processing the tensor.

-------------------------------------------------------------------------------
Examples:
-------------------------------------------------------------------------------

# Hugging Face model using the default MSE quantizer
python create_meta.py \
    --model bert-base-uncased \
    --out_dir ./bert_export

# Torchvision model with pretrained weights
python create_meta.py \
    --model resnet50 \
    --source torchvision \
    --weights ResNet50_Weights.DEFAULT \
    --out_dir ./resnet_export

# Torchvision model using RD quantization
python create_meta.py \
    --model resnet50 \
    --source torchvision \
    --weights ResNet50_Weights.DEFAULT \
    --out_dir ./resnet_rd_export \
    --quantizer rd \
    --lambda_rd 0.15

# Pre-quantized Torchvision model
python create_meta.py \
    --model resnet50 \
    --source torchvision \
    --weights ResNet50_QuantizedWeights.DEFAULT \
    --quantized \
    --out_dir ./resnet_quant_export

===============================================================================
"""

import os
import argparse
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
# may need to add more libs here for other models!


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
def optimal_uniform_quant(
            x,
            bitwidth,
            search_steps=40,
            mode="mse",
            lambda_rd=0.5,
    ):
        """
        Uniform quantizer optimized for MSE or rate-distortion.

        Modes
        -----
        mse
            Minimize reconstruction MSE only.

        rd
            Minimize:
                J = distortion + lambda * entropy

            where entropy approximates coding rate.
        """

        x = x.astype(np.float32)

        Qmax = (1 << (bitwidth - 1)) - 1

        if x.size == 0 or np.all(x == 0):
            return np.zeros_like(x, dtype=np.int32), 1.0


        std = float(np.std(x))

        if std == 0:
            return np.zeros_like(x, dtype=np.int32), 1.0


        # ----------------------------------------------------------
        # Search interval
        # ----------------------------------------------------------

        qstep_min = max(
            std / (1 << (bitwidth + 2)),
            1e-12
        )

        if mode == "mse":
            qstep_max = std * 4.0

        else:
            # Do not allow collapse into all zeros
            qstep_max = std * 4.0


        phi = (1 + np.sqrt(5.0)) / 2.0
        invphi = 1.0 / phi


        a = qstep_min
        b = qstep_max


        c = b - (b-a)*invphi
        d = a + (b-a)*invphi


        # ----------------------------------------------------------
        # RD cost function
        # ----------------------------------------------------------

        variance = np.mean(x*x) + 1e-12


        def cost(qstep):

            q = np.round(x / qstep)
            q = np.clip(q, -Qmax, Qmax)


            x_hat = q * qstep


            # ----------------------------
            # distortion
            # ----------------------------

            mse = np.mean((x - x_hat)**2)

            mse /= variance


            if mode == "mse":
                return mse


            # ----------------------------
            # entropy (rate)
            # ----------------------------

            q_int = q.astype(np.int32).ravel()


            hist = np.bincount(
                q_int + Qmax,
                minlength=2*Qmax+1
            )


            p = hist.astype(np.float64)

            p /= p.sum()

            p = p[p > 0]


            entropy = -(p*np.log2(p)).sum()


            # normalize to approximately [0,1]

            entropy /= bitwidth


            # ----------------------------
            # RD objective
            # ----------------------------

            return (
                mse +
                lambda_rd * entropy
            )


        # ----------------------------------------------------------
        # Golden section search
        # ----------------------------------------------------------

        fc = cost(c)
        fd = cost(d)


        for _ in range(search_steps):

            if fc < fd:

                b = d
                d = c
                fd = fc

                c = b - (b-a)*invphi
                fc = cost(c)

            else:

                a = c
                c = d
                fc = fd

                d = a + (b-a)*invphi
                fd = cost(d)


        qstep = (a+b)/2.0


        # ----------------------------------------------------------
        # Final quantization
        # ----------------------------------------------------------

        q = np.round(x / qstep)

        q = np.clip(
            q,
            -Qmax,
            Qmax
        )


        # ----------------------------------------------------------
        # Statistics
        # ----------------------------------------------------------

        q_int = q.astype(np.int32)

        hist = np.bincount(
            q_int.ravel()+Qmax,
            minlength=2*Qmax+1
        )

        p = hist.astype(np.float64)
        p /= p.sum()

        p = p[p>0]

        entropy = -(p*np.log2(p)).sum()


        print(
            f"step={qstep:.6e} "
            f"Entropy={entropy:.3f} "
            f"Zeros={100*np.mean(q_int==0):.2f}% "
            f"Mean|q|={np.mean(np.abs(q_int)):.3f} "
            f"MSE={np.mean((x-q_int*qstep)**2):.3e}"
        )


        return q_int, float(qstep)



# Core quantization entry point
# - Handles weights, biases, and buffers differently
# - Always outputs int32 (for C++ compatibility)
# - Bitwidth is used later for entropy coding, not storage
def quantize_tensor(arr, use_quant=True, tensor_kind ="weight", mode="mse", lambda_rd=0.5):
    numel = arr.size


    if tensor_kind == "buffer":
        return arr, 1.0, 32

    if not use_quant:
        # assume already quantized
        return arr, 1.0, 8


    #if numel < 32:
    #    bitwidth = 12
    #    qstep = np.max(np.abs(arr)) / (2**(bitwidth - 1) - 1 + 1e-8)
    #    q = np.round(arr / qstep)

    #elif tensor_kind == "weight":
    bitwidth = 8
    q, qstep = optimal_uniform_quant(arr, bitwidth, mode=mode, lambda_rd=lambda_rd)
   # else:
      #  bitwidth = 12
       # q, qstep = optimal_uniform_quant(arr, bitwidth)

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
# Flexible loader:
# - Torchvision: supports weights + quantized models
# - HuggingFace: tries causal LM → classifier → base model
# ============================================================

def load_model(name, source="hf", weights=None, quantized=False):
    if source == "torchvision":
        return load_torchvision_model(name, weights, quantized)

    print(f"Loading HuggingFace model: {name}")

    try:
        return AutoModelForCausalLM.from_pretrained(name)
    except Exception as e:
        print(e)
    

    try:
        return AutoModelForSequenceClassification.from_pretrained(name)
    except Exception as e:
        print(e)
        

    try:
        return AutoModel.from_pretrained(name)
    except Exception as e:
        print(e)

    raise RuntimeError(f"Could not load model: {name}")

def load_torchvision_model(model_name, weights_name=None, quantized=False):
    import importlib

    if quantized:
        models = importlib.import_module("torchvision.models.quantization")
    else:
        models = importlib.import_module("torchvision.models")

    print(f"Loading torchvision model: {model_name}")

    # Constructor
    if not hasattr(models, model_name):
        raise ValueError(
            f"Unknown torchvision model '{model_name}' "
            f"in module {models.__name__}"
        )

    model_fn = getattr(models, model_name)

    # Resolve weight enum without eval()
    weights = None
    if weights_name is not None:
        obj = models
        try:
            for part in weights_name.split("."):
                obj = getattr(obj, part)
            weights = obj
        except AttributeError:
            raise ValueError(
                f"Weight enum '{weights_name}' not found in "
                f"{models.__name__}"
            )

    kwargs = {}

    if weights is not None:
        kwargs["weights"] = weights

    if quantized:
        kwargs["quantize"] = True

    return model_fn(**kwargs)
# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--no_quant", action="store_true",
                        help="Skip quantization for already-quantized 8-bit parameter tensors")
    
    parser.add_argument("--source", default="hf",
                    choices=["hf", "torchvision"])

    parser.add_argument("--weights", default=None,
                        help="Torchvision weights enum (e.g., ResNet50_Weights.DEFAULT)")

    parser.add_argument("--quantized", action="store_true",
                        help="Load quantized torchvision model")

    parser.add_argument("--quantizer", choices=["mse","rd"], default="mse", help="Pure MSE minimization or rate-distortion")

    parser.add_argument("--lambda_rd", type=float, default = 0.15 , help="Lambda for rate-distortion, 0.0 = pure MSE, above=more compression")

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

        # If model is already quantized (e.g., torchvision quantized models),
        # use int_repr() to extract integer values directly
        if (args.quantized and hasattr(param, "int_repr")):
            arr = param.int_repr().cpu().numpy().astype(np.int32)
            q, qstep, bitwidth = quantize_tensor(
                arr,
                use_quant=False, # no quantization 
                tensor_kind=tensor_kind
            )
        else:
            arr = param.detach().to(torch.float32).cpu().numpy()
            q, qstep, bitwidth = quantize_tensor(
                arr,
                use_quant=True,
                tensor_kind=tensor_kind,
                mode=args.quantizer,
                lambda_rd=args.lambda_rd
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
    # Buffers are NOT part of named_parameters (e.g., BatchNorm stats)
    # They must NOT be quantized to preserve correctness
    # We store them as raw int32 with bitwidth=32
    for name, buf in tqdm(model.named_buffers(), desc="buffers"):
        arr = buf.detach().to(torch.float32).cpu().numpy()
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