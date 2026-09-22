#!/usr/bin/env python3

"""
StaticBAC reconstructed-model generator.

Reconstructs a neural-network model from StaticBAC decoded tensors.

Supported model sources:
    - torchvision
    - Hugging Face Transformers
    - auto detection

Supported reconstruction modes:
    --buffers original
        Reconstruct parameters and use original pretrained buffers.

    --buffers reconstructed
        Reconstruct parameters and buffers from StaticBAC decoded data.

    --buffers none
        Reconstruct parameters only and do not store buffers.

Typical usage
-------------

EfficientNet-B7:

python reconstruct_model.py \
    --source torchvision \
    --model efficientnet_b7 \
    --weights torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1 \
    --param-folder efficientnet_b7_decoded_015 \
    --param-meta models8_rd/efficientnet_b7/tensor.meta \
    --output efficientnet_b7_015_reconstructed.npz \
    --buffers original


BERT:

python reconstruct_model.py \
    --source hf \
    --model bert-base-uncased \
    --param-folder bert_decoded \
    --param-meta models8_rd/bert/tensor.meta \
    --output bert_reconstructed.npz \
    --buffers original


BERT with reconstructed buffers:

python reconstruct_model.py \
    --source hf \
    --model bert-base-uncased \
    --param-folder bert_decoded \
    --param-meta models8_rd/bert/tensor.meta \
    --buffer-folder bert_buffers_decoded \
    --buffer-meta models8_rd/bert_buffers/tensor.meta \
    --output bert_reconstructed_with_buffers.npz \
    --buffers reconstructed
"""

import argparse
import importlib
import os
import sys

import numpy as np
import torch


# ============================================================================
# Optional Hugging Face
# ============================================================================

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoModel,
    )

    TRANSFORMERS_AVAILABLE = True

except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ============================================================================
# StaticBAC bitwidth map
# ============================================================================

BITWIDTH_MAP = {
    0: 4,
    1: 8,
    2: 12,
    3: 16,
    4: 20,
    5: 24,
    6: 32,
}


# ============================================================================
# Model loader
# ============================================================================

def load_model(
    name,
    source="auto",
    weights=None,
    quantized=False,
):
    """
    Flexible model loader.

    source:
        auto
        torchvision
        hf
    """

    if source == "torchvision":
        return load_torchvision_model(
            name,
            weights,
            quantized,
        )

    if source == "hf":
        return load_huggingface_model(name)

    if source == "auto":

        # ------------------------------------------------------------
        # Try torchvision
        # ------------------------------------------------------------

        try:

            import torchvision.models as models

            if hasattr(models, name):

                print(
                    f"Auto-detected torchvision model: {name}"
                )

                return load_torchvision_model(
                    name,
                    weights,
                    quantized,
                )

        except Exception as e:

            print(
                f"torchvision detection failed: "
                f"{type(e).__name__}: {e}"
            )

        # ------------------------------------------------------------
        # Otherwise Hugging Face
        # ------------------------------------------------------------

        print(
            f"Auto-detected Hugging Face model: {name}"
        )

        return load_huggingface_model(name)

    raise ValueError(
        f"Unknown source '{source}'"
    )


def load_torchvision_model(
    model_name,
    weights_name=None,
    quantized=False,
):
    """
    Dynamically load a torchvision model.
    """

    if quantized:

        models = importlib.import_module(
            "torchvision.models.quantization"
        )

    else:

        models = importlib.import_module(
            "torchvision.models"
        )

    print(
        f"Loading torchvision model: {model_name}"
    )

    if not hasattr(models, model_name):

        raise ValueError(
            f"Unknown torchvision model '{model_name}' "
            f"in {models.__name__}"
        )

    model_fn = getattr(
        models,
        model_name,
    )

    weights = None

    if weights_name is not None:

        obj = models

        try:

            for part in weights_name.split("."):
                obj = getattr(obj, part)

            weights = obj

        except AttributeError:

            raise ValueError(
                f"Weight enum '{weights_name}' not found "
                f"in {models.__name__}"
            )

    kwargs = {}

    if weights is not None:
        kwargs["weights"] = weights

    if quantized:
        kwargs["quantize"] = True

    return model_fn(**kwargs)


def load_huggingface_model(name):
    """
    Flexible Hugging Face loader.

    Order:
        Causal LM
        Sequence classifier
        Base model
    """

    if not TRANSFORMERS_AVAILABLE:

        raise RuntimeError(
            "transformers is not installed.\n"
            "Install it with:\n"
            "pip install transformers"
        )

    print(
        f"Loading Hugging Face model: {name}"
    )

    # ------------------------------------------------------------
    # Causal LM
    # ------------------------------------------------------------

    try:

        model = AutoModelForCausalLM.from_pretrained(
            name
        )

        print(
            "Loaded as AutoModelForCausalLM"
        )

        return model

    except Exception as e:

        print(
            f"Causal LM loader failed: "
            f"{type(e).__name__}: {e}"
        )

    # ------------------------------------------------------------
    # Sequence classification
    # ------------------------------------------------------------

    try:

        model = AutoModelForSequenceClassification.from_pretrained(
            name
        )

        print(
            "Loaded as AutoModelForSequenceClassification"
        )

        return model

    except Exception as e:

        print(
            f"Sequence classifier loader failed: "
            f"{type(e).__name__}: {e}"
        )

    # ------------------------------------------------------------
    # Base model
    # ------------------------------------------------------------

    try:

        model = AutoModel.from_pretrained(
            name
        )

        print(
            "Loaded as AutoModel"
        )

        return model

    except Exception as e:

        print(
            f"Base model loader failed: "
            f"{type(e).__name__}: {e}"
        )

    raise RuntimeError(
        f"Could not load model: {name}"
    )


# ============================================================================
# Encoder metadata
# ============================================================================

def read_encoder_meta(path):

    qsteps = {}
    id_to_name = {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

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

        try:

            tensor_id = int(parts[0])
            name = parts[1]

            dims = int(parts[4])

            qstep_index = 5 + dims

            if qstep_index >= len(parts):

                raise RuntimeError(
                    f"Malformed encoder metadata line:\n{line}"
                )

            qstep = float(
                parts[qstep_index]
            )

        except ValueError as e:

            raise RuntimeError(
                f"Could not parse encoder metadata line:\n{line}"
            ) from e

        if tensor_id in id_to_name:

            raise RuntimeError(
                f"Duplicate tensor ID {tensor_id} "
                f"in encoder metadata."
            )

        id_to_name[tensor_id] = name
        qsteps[tensor_id] = qstep

    print(
        f"Loaded {len(id_to_name)} tensors "
        f"from encoder meta"
    )

    return qsteps, id_to_name


# ============================================================================
# Decoded metadata
# ============================================================================

def read_decoded_meta(path):

    tensors = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        lines = f.readlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if line.startswith("numTensors"):
            continue

        parts = line.split()

        if len(parts) < 5:
            continue

        try:

            idx = int(parts[0])
            filename = parts[1]
            bw_enum = int(parts[3])
            dims = int(parts[4])

        except ValueError as e:

            raise RuntimeError(
                f"Could not parse decoded metadata line:\n{line}"
            ) from e

        expected_fields = 5 + dims

        if len(parts) != expected_fields:

            raise RuntimeError(
                f"Malformed decoded metadata line:\n"
                f"{line}\n"
                f"Expected {expected_fields} fields, "
                f"got {len(parts)}"
            )

        if bw_enum not in BITWIDTH_MAP:

            raise RuntimeError(
                f"Unknown bitwidth enum {bw_enum} "
                f"in line:\n{line}"
            )

        shape = tuple(
            map(
                int,
                parts[5:5 + dims]
            )
        )

        tensors.append(
            {
                "idx": idx,
                "filename": filename,
                "bitwidth": BITWIDTH_MAP[bw_enum],
                "shape": shape,
            }
        )

    return tensors


# ============================================================================
# Load decoded tensor
# ============================================================================

def load_tensor(
    path,
    shape,
):

    arr = np.fromfile(
        path,
        dtype=np.int32,
    )

    expected = int(
        np.prod(shape)
    )

    if arr.size != expected:

        raise RuntimeError(
            f"{path}: expected {expected} values "
            f"but found {arr.size}"
        )

    return arr.reshape(shape)


# ============================================================================
# Model state
# ============================================================================

def get_model_parameters(model):

    return dict(
        model.named_parameters()
    )


def get_model_buffers(model):

    return dict(
        model.named_buffers()
    )


def print_model_summary(model):

    parameters = get_model_parameters(model)
    buffers = get_model_buffers(model)

    parameter_elements = sum(
        tensor.numel()
        for tensor in parameters.values()
    )

    buffer_elements = sum(
        tensor.numel()
        for tensor in buffers.values()
    )

    print("\n" + "=" * 80)
    print("MODEL STATE")
    print("=" * 80)

    print(
        f"Parameters:          {len(parameters)}"
    )

    print(
        f"Buffers:             {len(buffers)}"
    )

    print(
        f"Total state tensors: "
        f"{len(parameters) + len(buffers)}"
    )

    print(
        f"Parameter elements:  "
        f"{parameter_elements:,}"
    )

    print(
        f"Buffer elements:     "
        f"{buffer_elements:,}"
    )


# ============================================================================
# Classify decoded tensor
# ============================================================================

def classify_tensor(
    name,
    parameters,
    buffers,
):
    """
    Determine whether a decoded tensor is a model parameter or buffer.

    Returns:
        "parameter"
        "buffer"
        "unknown"
    """

    if name in parameters:
        return "parameter"

    if name in buffers:
        return "buffer"

    return "unknown"


# ============================================================================
# Reconstruct one tensor
# ============================================================================

def reconstruct_tensor(
    tensor_info,
    folder,
    tensor_name,
    qsteps,
    model_tensor,
    apply_qstep,
):
    """
    Load one decoded tensor and convert it back to float32 model values.
    """

    tensor_id = tensor_info["idx"]
    filename = tensor_info["filename"]
    bitwidth = tensor_info["bitwidth"]
    shape = tensor_info["shape"]

    bin_path = os.path.join(
        folder,
        filename,
    )

    if not os.path.isfile(bin_path):

        raise FileNotFoundError(
            f"Decoded tensor file not found:\n"
            f"  {bin_path}\n"
            f"  Tensor: {tensor_name}\n"
            f"  ID:     {tensor_id}"
        )

    tensor = load_tensor(
        bin_path,
        shape,
    ).astype(
        np.float32
    )

    # ------------------------------------------------------------
    # Check decoded shape against actual model
    # ------------------------------------------------------------

    expected_shape = tuple(
        model_tensor.shape
    )

    if tuple(tensor.shape) != expected_shape:

        raise RuntimeError(
            f"Shape mismatch for tensor '{tensor_name}':\n"
            f"  Model:       {expected_shape}\n"
            f"  Decoded:     {tensor.shape}\n"
            f"  Tensor ID:   {tensor_id}\n"
            f"  File:        {bin_path}"
        )

    # ------------------------------------------------------------
    # Apply inverse quantization
    # ------------------------------------------------------------

    qstep = qsteps.get(
        tensor_id,
        None,
    )

    if apply_qstep and bitwidth != 32:

        if qstep is None:

            raise RuntimeError(
                f"No qstep available for quantized tensor:\n"
                f"  ID:       {tensor_id}\n"
                f"  Name:     {tensor_name}\n"
                f"  Bitwidth: {bitwidth}"
            )

        tensor *= qstep

    return tensor, qstep


# ============================================================================
# Build reconstruction
# ============================================================================

def build_reconstruction(
    decoded_meta,
    folder,
    qsteps,
    id_to_name,
    model,
    buffer_mode="original",
    apply_qstep=True,
):
    """
    Reconstruct the model from ONE decoded metadata file.

    The metadata can contain both parameters and buffers.

    Parameters are always reconstructed.

    Buffers:
        original       -> copy original model buffers
        reconstructed  -> reconstruct buffers from decoded binaries
        none           -> don't include buffers in output
    """

    parameters = get_model_parameters(
        model
    )

    buffers = get_model_buffers(
        model
    )

    # ------------------------------------------------------------------------
    # Basic metadata consistency
    # ------------------------------------------------------------------------

    decoded_ids = {
        tensor["idx"]
        for tensor in decoded_meta
    }

    if len(decoded_ids) != len(decoded_meta):

        raise RuntimeError(
            "Duplicate tensor IDs found in decoded_tensors.meta."
        )

    missing_names = []

    for tensor_info in decoded_meta:

        tensor_id = tensor_info["idx"]

        if tensor_id not in id_to_name:

            missing_names.append(
                tensor_id
            )

    if missing_names:

        raise RuntimeError(
            "Decoded metadata contains tensor IDs that are "
            "missing from encoder metadata:\n"
            + "\n".join(
                f"  {x}"
                for x in missing_names[:30]
            )
        )

    # ------------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------------

    parameter_tensors = []
    buffer_tensors = []
    unknown_tensors = []

    for tensor_info in decoded_meta:

        tensor_id = tensor_info["idx"]

        name = id_to_name[tensor_id]

        category = classify_tensor(
            name,
            parameters,
            buffers,
        )

        if category == "parameter":

            parameter_tensors.append(
                (tensor_info, name)
            )

        elif category == "buffer":

            buffer_tensors.append(
                (tensor_info, name)
            )

        else:

            unknown_tensors.append(
                (tensor_info, name)
            )

    # ------------------------------------------------------------------------
    # Print classification
    # ------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("DECODED TENSOR CLASSIFICATION")
    print("=" * 80)

    print(
        f"Decoded metadata tensors: "
        f"{len(decoded_meta)}"
    )

    print(
        f"Parameters in metadata:   "
        f"{len(parameter_tensors)}"
    )

    print(
        f"Buffers in metadata:      "
        f"{len(buffer_tensors)}"
    )

    print(
        f"Unknown tensors:           "
        f"{len(unknown_tensors)}"
    )

    print(
        f"Model parameters:          "
        f"{len(parameters)}"
    )

    print(
        f"Model buffers:             "
        f"{len(buffers)}"
    )

    # ------------------------------------------------------------------------
    # Unknown tensors are probably a metadata/model mismatch.
    # ------------------------------------------------------------------------

    if unknown_tensors:

        print(
            "\nUnknown decoded tensors:"
        )

        for tensor_info, name in unknown_tensors[:50]:

            print(
                f"  ID {tensor_info['idx']}: "
                f"{name}"
            )

        if len(unknown_tensors) > 50:

            print(
                f"  ... and "
                f"{len(unknown_tensors) - 50} more"
            )

        raise RuntimeError(
            f"{len(unknown_tensors)} decoded tensors do not "
            "correspond to parameters or buffers of the loaded model."
        )

    # ------------------------------------------------------------------------
    # Parameter coverage
    # ------------------------------------------------------------------------

    decoded_parameter_names = {
        name
        for _, name in parameter_tensors
    }

    missing_parameters = sorted(
        set(parameters.keys())
        - decoded_parameter_names
    )

    if missing_parameters:

        print(
            "\nMissing model parameters:"
        )

        for name in missing_parameters[:50]:
            print(
                f"  {name}"
            )

        raise RuntimeError(
            f"{len(missing_parameters)} model parameters "
            "are missing from decoded_tensors.meta."
        )

    # ------------------------------------------------------------------------
    # Reconstruct parameters
    # ------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("RECONSTRUCTING PARAMETERS")
    print("=" * 80)

    parameter_data = {}

    for tensor_info, name in parameter_tensors:

        tensor, qstep = reconstruct_tensor(
            tensor_info=tensor_info,
            folder=folder,
            tensor_name=name,
            qsteps=qsteps,
            model_tensor=parameters[name],
            apply_qstep=apply_qstep,
        )

        tensor_id = tensor_info["idx"]

        key = (
            f"param_{tensor_id:03d}_{name}"
        )

        if key in parameter_data:

            raise RuntimeError(
                f"Duplicate output key:\n{key}"
            )

        parameter_data[key] = tensor

        print(
            f"\n[PARAM ID {tensor_id}] {name}"
        )

        print(
            f"    shape={tensor.shape} "
            f"bitwidth={tensor_info['bitwidth']}"
        )

        if qstep is not None:

            print(
                f"    qstep={qstep}"
            )

    # ------------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------------

    buffer_data = {}

    if buffer_mode == "none":

        print("\n" + "=" * 80)
        print("BUFFERS")
        print("=" * 80)

        print(
            "Buffer reconstruction disabled."
        )

        print(
            f"Ignoring {len(buffer_tensors)} "
            "buffer tensors from decoded metadata."
        )

    elif buffer_mode == "original":

        print("\n" + "=" * 80)
        print("BUFFERS")
        print("=" * 80)

        if not buffers:

            print(
                "Model has no buffers."
            )

        else:

            print(
                "Using original pretrained buffers."
            )

            print(
                f"Ignoring {len(buffer_tensors)} "
                "decoded buffer tensors."
            )

            for i, (name, buf) in enumerate(
                buffers.items()
            ):

                key = (
                    f"buffer_{i:03d}_{name}"
                )

                buffer_data[key] = (
                    buf.detach()
                    .cpu()
                    .numpy()
                )

    elif buffer_mode == "reconstructed":

        print("\n" + "=" * 80)
        print("RECONSTRUCTING BUFFERS")
        print("=" * 80)

        if not buffers:

            print(
                "Model has no buffers."
            )

        else:

            decoded_buffer_names = {
                name
                for _, name in buffer_tensors
            }

            missing_buffers = sorted(
                set(buffers.keys())
                - decoded_buffer_names
            )

            # num_batches_tracked is a runtime counter and can
            # legitimately be absent from a reconstruction.
            required_buffers = sorted(
                name
                for name in buffers
                if not name.endswith(
                    "num_batches_tracked"
                )
            )

            missing_required = [
                name
                for name in required_buffers
                if name not in decoded_buffer_names
            ]

            if missing_required:

                print(
                    "\nMissing reconstructed buffers:"
                )

                for name in missing_required[:50]:
                    print(
                        f"  {name}"
                    )

                raise RuntimeError(
                    f"{len(missing_required)} required buffers "
                    "are missing from decoded_tensors.meta."
                )

            print(
                f"Reconstructing "
                f"{len(buffer_tensors)} decoded buffers."
            )

            for tensor_info, name in buffer_tensors:

                # Runtime counters are deliberately kept from
                # the original model.
                if name.endswith(
                    "num_batches_tracked"
                ):

                    print(
                        f"\n[BUFFER ID {tensor_info['idx']}] "
                        f"{name}"
                    )

                    print(
                        "    num_batches_tracked -> "
                        "keeping original model buffer"
                    )

                    continue

                tensor, qstep = reconstruct_tensor(
                    tensor_info=tensor_info,
                    folder=folder,
                    tensor_name=name,
                    qsteps=qsteps,
                    model_tensor=buffers[name],
                    apply_qstep=apply_qstep,
                )

                tensor_id = tensor_info["idx"]

                key = (
                    f"buffer_{tensor_id:03d}_{name}"
                )

                buffer_data[key] = tensor

                print(
                    f"\n[BUFFER ID {tensor_id}] {name}"
                )

                print(
                    f"    shape={tensor.shape} "
                    f"bitwidth={tensor_info['bitwidth']}"
                )

                if qstep is not None:

                    print(
                        f"    qstep={qstep}"
                    )

    else:

        raise ValueError(
            f"Unknown buffer mode: {buffer_mode}"
        )

    return parameter_data, buffer_data


# ============================================================================
# Summary
# ============================================================================

def print_summary(
    parameter_data,
    buffer_data,
    decoded_count,
):

    parameter_elements = sum(
        tensor.size
        for tensor in parameter_data.values()
    )

    buffer_elements = sum(
        tensor.size
        for tensor in buffer_data.values()
    )

    print("\n" + "=" * 80)
    print("RECONSTRUCTION SUMMARY")
    print("=" * 80)

    print(
        f"Decoded metadata tensors: "
        f"{decoded_count}"
    )

    print(
        f"Output parameter tensors: "
        f"{len(parameter_data)}"
    )

    print(
        f"Output buffer tensors:    "
        f"{len(buffer_data)}"
    )

    print(
        f"Output total tensors:     "
        f"{len(parameter_data) + len(buffer_data)}"
    )

    print(
        f"Parameter elements:       "
        f"{parameter_elements:,}"
    )

    print(
        f"Buffer elements:          "
        f"{buffer_elements:,}"
    )


# ============================================================================
# Main
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate a StaticBAC reconstructed-model NPZ "
            "from decoded tensors."
        )
    )

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model name. Examples:\n"
            "  efficientnet_b7\n"
            "  resnet50\n"
            "  vit_b_16\n"
            "  bert-base-uncased\n"
            "  openai-community/gpt2"
        ),
    )

    parser.add_argument(
        "--source",
        choices=[
            "auto",
            "torchvision",
            "hf",
        ],
        default="auto",
    )

    parser.add_argument(
        "--weights",
        default=None,
        help=(
            "Torchvision weight enum, e.g. "
            "torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1"
        ),
    )

    parser.add_argument(
        "--quantized",
        action="store_true",
    )

    # ------------------------------------------------------------------------
    # StaticBAC data
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--decoded-folder",
        required=True,
        help=(
            "Folder containing decoded .bin files and "
            "decoded_tensors.meta."
        ),
    )

    parser.add_argument(
        "--encoder-meta",
        required=True,
        help=(
            "Encoder tensor.meta corresponding to the "
            "decoded StaticBAC stream."
        ),
    )

    # ------------------------------------------------------------------------
    # Buffer handling
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--buffers",
        choices=[
            "none",
            "original",
            "reconstructed",
        ],
        default="none",
        help=(
            "Buffer handling:\n"
            "  none           do not store buffers\n"
            "  original       store original pretrained buffers\n"
            "  reconstructed  reconstruct buffers from decoded data\n"
            "Default: none"
        ),
    )

    # ------------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--no-qstep",
        action="store_true",
        help=(
            "Do not apply inverse quantization qsteps."
        ),
    )

    # ------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--output",
        required=True,
        help="Output NPZ filename.",
    )

    args = parser.parse_args()

    # =========================================================================
    # Header
    # =========================================================================

    print("=" * 80)
    print("STATICBAC RECONSTRUCTED MODEL GENERATOR")
    print("=" * 80)

    print(
        f"Model:          {args.model}"
    )

    print(
        f"Source:          {args.source}"
    )

    print(
        f"Decoded folder:  {args.decoded_folder}"
    )

    print(
        f"Encoder meta:    {args.encoder_meta}"
    )

    print(
        f"Buffers:         {args.buffers}"
    )

    print(
        f"Apply qsteps:    {not args.no_qstep}"
    )

    print(
        f"Output:          {args.output}"
    )

    # =========================================================================
    # Load model
    # =========================================================================

    print("\n" + "=" * 80)
    print("LOADING PRETRAINED MODEL")
    print("=" * 80)

    model = load_model(
        name=args.model,
        source=args.source,
        weights=args.weights,
        quantized=args.quantized,
    )

    model.eval()

    print_model_summary(
        model
    )

    # =========================================================================
    # Read metadata
    # =========================================================================

    print("\n" + "=" * 80)
    print("READING STATICBAC METADATA")
    print("=" * 80)

    qsteps, id_to_name = read_encoder_meta(
        args.encoder_meta
    )

    decoded_meta_path = os.path.join(
        args.decoded_folder,
        "decoded_tensors.meta",
    )

    if not os.path.isfile(
        decoded_meta_path
    ):

        raise FileNotFoundError(
            f"decoded_tensors.meta not found:\n"
            f"{decoded_meta_path}"
        )

    decoded_meta = read_decoded_meta(
        decoded_meta_path
    )

    print(
        f"Decoded tensors: "
        f"{len(decoded_meta)}"
    )

    # =========================================================================
    # Reconstruct
    # =========================================================================

    parameter_data, buffer_data = build_reconstruction(
        decoded_meta=decoded_meta,
        folder=args.decoded_folder,
        qsteps=qsteps,
        id_to_name=id_to_name,
        model=model,
        buffer_mode=args.buffers,
        apply_qstep=not args.no_qstep,
    )

    # =========================================================================
    # Summary
    # =========================================================================

    print_summary(
        parameter_data,
        buffer_data,
        len(decoded_meta),
    )

    # =========================================================================
    # Merge
    # =========================================================================

    all_data = {}

    all_data.update(
        parameter_data
    )

    all_data.update(
        buffer_data
    )

    # =========================================================================
    # Save
    # =========================================================================

    output_path = os.path.abspath(
        args.output
    )

    output_dir = os.path.dirname(
        output_path
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    print("\n" + "=" * 80)
    print("SAVING NPZ")
    print("=" * 80)

    np.savez(
        output_path,
        **all_data,
    )

    print(
        f"Saved reconstructed model:"
    )

    print(
        f"  {output_path}"
    )

    print(
        f"\nNPZ entries: "
        f"{len(all_data)}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()