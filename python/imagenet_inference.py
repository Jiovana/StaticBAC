import os
import re
import argparse
import numpy as np
import torch
import torchvision.models as tvmodels
from torchvision.models import (
    ResNet50_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B7_Weights,
    ViT_B_16_Weights,
)
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_MODEL_NAME = "resnet50"

IMAGENET_VAL_DIR = r"C:\Users\Jiovana\Documents\imagenet_validation"

# NPZ files are expected in the StaticBAC root folder, where the script runs.
DEFAULT_RECONSTRUCTION_DIR = "."

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_CONFIGS = {
    "resnet50": {
        "weights": ResNet50_Weights.IMAGENET1K_V1,
        "batch_size": 128,
    },
    "efficientnet_b0": {
        "weights": EfficientNet_B0_Weights.IMAGENET1K_V1,
        "batch_size": 128,
    },
    "efficientnet_b7": {
        "weights": EfficientNet_B7_Weights.IMAGENET1K_V1,
        "batch_size": 16,
    },
    "vit_b_16": {
        "weights": ViT_B_16_Weights.IMAGENET1K_V1,
        "batch_size": 64,
    },
}


# Known original accuracies.
# Add other models here if you want to run them without explicitly
# providing --original-top1 / --original-top5.
DEFAULT_ORIGINAL_ACCURACY = {
    "efficientnet_b7": {
        "top1": 84.114,
        "top5": 96.904,
    },
   "vit_b_16": {
       "top1": 81.07,
       "top5": 95.32,
   },
   "resnet50": {
       "top1": 76.14,
       "top5": 92.87,
   },
}


# ============================================================================
# Model loading
# ============================================================================

def load_model(model_name):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unsupported model: {model_name}\n"
            f"Supported models: {list(MODEL_CONFIGS.keys())}"
        )

    weights = MODEL_CONFIGS[model_name]["weights"]

    if model_name == "resnet50":
        model = tvmodels.resnet50(weights=weights)

    elif model_name == "efficientnet_b0":
        model = tvmodels.efficientnet_b0(weights=weights)

    elif model_name == "efficientnet_b7":
        model = tvmodels.efficientnet_b7(weights=weights)

    elif model_name == "vit_b_16":
        model = tvmodels.vit_b_16(weights=weights)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def recommend_batch_size(model_name):
    return MODEL_CONFIGS[model_name]["batch_size"]


# ============================================================================
# Reconstruction file discovery
# ============================================================================

def find_reconstruction_file(model_name, reconstruction_dir):
    """
    Find the StaticBAC reconstruction NPZ.

    Priority:
      1. <model>_lambda015_reconstructed.npz
      2. <model>_mse_reconstructed.npz
      3. <model>_rd_reconstructed.npz
      4. <model>_rec_params.npz
      5. Any <model>*reconstructed.npz
    """

    reconstruction_dir = os.path.abspath(reconstruction_dir)

    preferred_names = [
        f"{model_name}_lambda015_reconstructed.npz",
        f"{model_name}_mse_reconstructed.npz",
        f"{model_name}_rd_reconstructed.npz",
        f"{model_name}_rec_params.npz",
    ]

    # First look for exact/preferred names.
    for name in preferred_names:
        candidate = os.path.join(reconstruction_dir, name)
        if os.path.isfile(candidate):
            return candidate

    # Then search recursively.
    candidates = []

    for root, _, files in os.walk(reconstruction_dir):
        for filename in files:
            lower = filename.lower()

            if (
                lower.endswith(".npz")
                and model_name.lower() in lower
                and "reconstructed" in lower
            ):
                candidates.append(os.path.join(root, filename))

    if not candidates:
        raise FileNotFoundError(
            f"No StaticBAC reconstruction found for '{model_name}' "
            f"under:\n{reconstruction_dir}"
        )

    candidates.sort()

    print("\nCandidate reconstruction files:")
    for candidate in candidates:
        print(f"  {candidate}")

    print(f"\nUsing: {candidates[0]}")

    return candidates[0]


# ============================================================================
# Model state helpers
# ============================================================================

def get_model_parameters(model):
    return dict(model.named_parameters())


def get_model_buffers(model):
    return dict(model.named_buffers())


def get_model_state(model):
    """
    Return a detached CPU copy of the complete model state.
    """
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


# ============================================================================
# StaticBAC NPZ key handling
# ============================================================================

# StaticBAC currently exports keys such as:
#
#   param_000_features.0.0.weight
#   param_001_features.0.1.weight
#
# and, unfortunately, also:
#
#   buffer_000_features.0.1.running_mean
#   buffer_001_features.0.1.running_var
#
# We intentionally keep ONLY parameter entries.


PARAM_KEY_RE = re.compile(r"^param_(\d+)_(.+)$")
BUFFER_KEY_RE = re.compile(r"^buffer_(\d+)_(.+)$")


def classify_reconstruction_key(key):
    """
    Classify a raw StaticBAC NPZ key.

    Returns:
        ("parameter", actual_name)
        ("buffer", actual_name)
        ("other", key)
    """

    match = PARAM_KEY_RE.match(key)

    if match:
        return "parameter", match.group(2)

    match = BUFFER_KEY_RE.match(key)

    if match:
        return "buffer", match.group(2)

    return "other", key


# ============================================================================
# Reconstruction loading
# ============================================================================

def load_reconstruction(npz_path):
    """
    Load a StaticBAC reconstruction.

    IMPORTANT:
    StaticBAC NPZ files may contain both parameters and buffers.

    For inference we intentionally ignore buffers. Only entries with
    the form:

        param_<index>_<actual_parameter_name>

    are loaded.

    The original model buffers therefore remain untouched.
    """

    print("\n" + "=" * 80)
    print("LOADING STATICBAC RECONSTRUCTION")
    print("=" * 80)
    print(f"File: {os.path.basename(npz_path)}")

    raw_data = {}

    with np.load(npz_path, allow_pickle=False) as data:
        for key in data.files:
            raw_data[key] = np.ascontiguousarray(data[key])

    print(f"NPZ entries: {len(raw_data)}")

    parameter_data = {}
    buffer_entries = {}
    other_entries = {}

    for raw_key, value in raw_data.items():

        kind, actual_name = classify_reconstruction_key(raw_key)

        if kind == "parameter":
            if actual_name in parameter_data:
                raise RuntimeError(
                    "Duplicate parameter reconstruction key:\n"
                    f"  Raw key:        {raw_key}\n"
                    f"  Parameter name: {actual_name}"
                )

            parameter_data[actual_name] = value

        elif kind == "buffer":
            # Intentionally ignored.
            buffer_entries[actual_name] = value

        else:
            other_entries[actual_name] = value

    print(f"Parameter entries kept: {len(parameter_data)}")
    print(f"Buffer entries ignored: {len(buffer_entries)}")
    print(f"Other entries ignored: {len(other_entries)}")

    print("\nFirst reconstructed parameters:")

    for i, (name, value) in enumerate(parameter_data.items()):
        if i >= 10:
            break

        print(
            f"  {name}: "
            f"shape={value.shape}, "
            f"dtype={value.dtype}"
        )

    return parameter_data


# ============================================================================
# Coverage check
# ============================================================================

def check_coverage(model, reconstruction):
    """
    Check whether all model parameters are present.

    Buffers are intentionally NOT required because StaticBAC inference
    uses the original pretrained buffers.
    """

    print("\n" + "=" * 80)
    print("STATICBAC PARAMETER COVERAGE CHECK")
    print("=" * 80)

    model_parameters = get_model_parameters(model)
    model_buffers = get_model_buffers(model)

    reconstructed_names = set(reconstruction.keys())
    parameter_names = set(model_parameters.keys())

    reconstructed_parameters = sorted(
        reconstructed_names & parameter_names
    )

    missing_parameters = sorted(
        parameter_names - reconstructed_names
    )

    unexpected_parameters = sorted(
        reconstructed_names - parameter_names
    )

    print(f"Model parameters:             {len(model_parameters)}")
    print(f"Model buffers:                {len(model_buffers)}")
    print(f"Reconstructed parameters:     {len(reconstructed_parameters)}")
    print(f"Missing parameters:           {len(missing_parameters)}")
    print(f"Unexpected reconstruction:    {len(unexpected_parameters)}")

    if missing_parameters:
        print("\nMissing parameters:")

        for name in missing_parameters[:30]:
            print(f"  {name}")

        if len(missing_parameters) > 30:
            print(
                f"  ... and {len(missing_parameters) - 30} more"
            )

        raise RuntimeError(
            f"{len(missing_parameters)} model parameters are missing "
            "from the StaticBAC reconstruction."
        )

    # Check shapes.
    shape_errors = []

    for name in reconstructed_parameters:
        expected = tuple(model_parameters[name].shape)
        actual = tuple(reconstruction[name].shape)

        if expected != actual:
            shape_errors.append(
                (name, expected, actual)
            )

    if shape_errors:
        print("\nShape mismatches:")

        for name, expected, actual in shape_errors[:30]:
            print(
                f"  {name}: "
                f"model={expected}, reconstruction={actual}"
            )

        raise RuntimeError(
            f"{len(shape_errors)} parameter tensors have shape mismatches."
        )

    print("\nParameter coverage: OK")
    print("Buffers: ORIGINAL MODEL BUFFERS WILL BE USED")

    if unexpected_parameters:
        print(
            "\nNote: the NPZ contains additional entries that do not "
            "correspond to model parameters. They are ignored."
        )

        for name in unexpected_parameters[:20]:
            print(f"  {name}")

        if len(unexpected_parameters) > 20:
            print(
                f"  ... and {len(unexpected_parameters) - 20} more"
            )

    return True


# ============================================================================
# Parameter loading
# ============================================================================

def load_reconstructed_parameters_only(model, reconstruction):
    """
    Replace model parameters with StaticBAC reconstructed parameters.

    All model buffers remain exactly as loaded from the pretrained model.
    """

    model_parameters = get_model_parameters(model)

    loaded = 0

    with torch.no_grad():

        for name, value in reconstruction.items():

            if name not in model_parameters:
                continue

            target = model_parameters[name]

            tensor = torch.from_numpy(value)

            # Make sure the tensor has the same dtype as the model.
            tensor = tensor.to(
                dtype=target.dtype,
                device=target.device
            )

            target.copy_(tensor)

            loaded += 1

    print(
        f"Loaded {loaded}/{len(model_parameters)} "
        "reconstructed parameters."
    )

    if loaded != len(model_parameters):
        raise RuntimeError(
            f"Only {loaded}/{len(model_parameters)} model parameters "
            "were loaded."
        )

    return model


# ============================================================================
# Verify loaded parameters
# ============================================================================

def verify_loaded_parameters(model, reconstruction):
    """
    Verify that every reconstructed parameter was actually loaded.
    """

    model_parameters = get_model_parameters(model)

    errors = []

    for name, value in reconstruction.items():

        if name not in model_parameters:
            continue

        expected = torch.from_numpy(value).to(
            dtype=model_parameters[name].dtype,
            device=model_parameters[name].device,
        )

        actual = model_parameters[name]

        if expected.shape != actual.shape:
            errors.append(
                f"{name}: shape mismatch"
            )
            continue

        if not torch.equal(actual, expected):
            errors.append(
                f"{name}: tensor contents differ"
            )

    if errors:

        print("\nParameter verification errors:")

        for error in errors[:20]:
            print(f"  {error}")

        raise RuntimeError(
            f"{len(errors)} reconstructed parameters failed verification."
        )

    print("Reconstructed parameter verification: OK")


# ============================================================================
# Reconstruction error analysis
# ============================================================================

def analyze_reconstruction(model, original_state, reconstruction):
    """
    Compare the reconstructed parameters against the original model.

    Only model parameters are analyzed because buffers are intentionally
    not reconstructed by StaticBAC.
    """

    print("\n" + "=" * 80)
    print("STATICBAC PARAMETER RECONSTRUCTION ERROR")
    print("=" * 80)

    model_parameters = get_model_parameters(model)

    total_elements = 0
    total_changed = 0

    sum_abs = 0.0
    sum_sq = 0.0
    max_abs = 0.0

    tensor_stats = []

    for name in model_parameters:

        if name not in reconstruction:
            continue

        original = original_state[name].float()

        reconstructed = torch.from_numpy(
            reconstruction[name]
        ).float()

        diff = reconstructed - original
        abs_diff = diff.abs()

        n = diff.numel()

        total_elements += n
        total_changed += int(torch.count_nonzero(diff))

        sum_abs += float(abs_diff.sum())
        sum_sq += float((diff * diff).sum())

        tensor_max = float(abs_diff.max()) if n > 0 else 0.0
        max_abs = max(max_abs, tensor_max)

        tensor_stats.append(
            {
                "name": name,
                "mean_abs": float(abs_diff.mean()),
                "rmse": float(torch.sqrt(torch.mean(diff * diff))),
                "max_abs": tensor_max,
                "changed": int(torch.count_nonzero(diff)),
                "elements": n,
            }
        )

    if total_elements == 0:
        print("No reconstructed parameters available for analysis.")
        return

    global_mean_abs = sum_abs / total_elements
    global_rmse = (sum_sq / total_elements) ** 0.5
    changed_ratio = total_changed / total_elements

    print(f"Parameters analyzed: {len(tensor_stats)}")
    print(f"Elements analyzed:   {total_elements:,}")
    print(f"Mean absolute error: {global_mean_abs:.8e}")
    print(f"RMSE:                {global_rmse:.8e}")
    print(f"Maximum absolute:    {max_abs:.8e}")
    print(
        f"Changed elements:    "
        f"{total_changed:,} "
        f"({100.0 * changed_ratio:.4f}%)"
    )

    tensor_stats.sort(
        key=lambda x: x["mean_abs"],
        reverse=True
    )

    print("\nLargest mean-absolute errors:")

    for stat in tensor_stats[:20]:

        print(
            f"  {stat['name']}: "
            f"MAE={stat['mean_abs']:.4e}, "
            f"RMSE={stat['rmse']:.4e}, "
            f"MAX={stat['max_abs']:.4e}, "
            f"changed={stat['changed']:,}/{stat['elements']:,}"
        )


# ============================================================================
# ImageNet evaluation
# ============================================================================

def evaluate_imagenet(
    model,
    dataset,
    batch_size,
    num_workers=8,
):
    """
    Evaluate ImageNet top-1 and top-5 accuracy.
    """

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": DEVICE.type == "cuda",
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    loader = DataLoader(
        dataset,
        **loader_kwargs,
    )

    model.eval()
    model.to(DEVICE)

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    print("\n" + "=" * 80)
    print("IMAGENET INFERENCE")
    print("=" * 80)

    with torch.inference_mode():

        for images, targets in tqdm(
            loader,
            desc="Inference",
            unit="batch",
        ):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            _, top5 = torch.topk(
                outputs,
                k=5,
                dim=1
            )

            correct = top5.eq(
                targets.view(-1, 1)
            )

            correct_top1 += correct[:, 0].sum().item()
            correct_top5 += correct.any(dim=1).sum().item()

            total += targets.size(0)

    top1 = 100.0 * correct_top1 / total
    top5 = 100.0 * correct_top5 / total

    print(f"\nImages: {total:,}")
    print(f"Top-1:  {top1:.3f}%")
    print(f"Top-5:  {top5:.3f}%")

    return top1, top5


# ============================================================================
# Accuracy handling
# ============================================================================

def get_original_accuracy(model_name, args):

    if args.original_top1 is not None and args.original_top5 is not None:
        return args.original_top1, args.original_top5

    if model_name in DEFAULT_ORIGINAL_ACCURACY:

        values = DEFAULT_ORIGINAL_ACCURACY[model_name]

        return values["top1"], values["top5"]

    raise RuntimeError(
        f"No default original accuracy is available for '{model_name}'.\n"
        "Provide:\n"
        f"  --original-top1 <value> --original-top5 <value>"
    )


# ============================================================================
# Main
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate StaticBAC reconstructed vision models "
            "on ImageNet."
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        choices=list(MODEL_CONFIGS.keys()),
    )

    parser.add_argument(
        "--reconstruction-dir",
        default=DEFAULT_RECONSTRUCTION_DIR,
        help=(
            "Directory containing the StaticBAC NPZ reconstruction. "
            "Default: current working directory."
        ),
    )

    parser.add_argument(
        "--reconstruction-file",
        default=None,
        help="Explicit NPZ reconstruction filename/path.",
    )

    parser.add_argument(
        "--original-top1",
        type=float,
        default=None,
        help="Original pretrained model top-1 accuracy.",
    )

    parser.add_argument(
        "--original-top5",
        type=float,
        default=None,
        help="Original pretrained model top-5 accuracy.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the recommended batch size.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of ImageNet DataLoader workers.",
    )

    args = parser.parse_args()

    model_name = args.model

    print("=" * 80)
    print("STATICBAC VISION MODEL RECONSTRUCTION EVALUATION")
    print("=" * 80)

    print(f"Model:       {model_name}")
    print(f"Device:      {DEVICE}")

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else recommend_batch_size(model_name)
    )

    print(f"Batch size:  {batch_size}")
    print(f"Workers:     {args.workers}")

    # ------------------------------------------------------------------------
    # Original accuracy
    # ------------------------------------------------------------------------

    original_top1, original_top5 = get_original_accuracy(
        model_name,
        args,
    )

    print(
        f"Original top-1: {original_top1:.3f}%"
    )

    print(
        f"Original top-5: {original_top5:.3f}%"
    )

    # ------------------------------------------------------------------------
    # Find reconstruction
    # ------------------------------------------------------------------------

    if args.reconstruction_file is not None:

        reconstruction_path = args.reconstruction_file

        if not os.path.isabs(reconstruction_path):
            reconstruction_path = os.path.join(
                args.reconstruction_dir,
                reconstruction_path,
            )

        if not os.path.isfile(reconstruction_path):
            raise FileNotFoundError(
                f"Reconstruction file not found:\n"
                f"{reconstruction_path}"
            )

    else:

        reconstruction_path = find_reconstruction_file(
            model_name,
            args.reconstruction_dir,
        )

    print(
        f"Reconstruction: "
        f"{os.path.basename(reconstruction_path)}"
    )

    # ------------------------------------------------------------------------
    # Load original model
    # ------------------------------------------------------------------------

    print("\nLoading original pretrained model...")

    original_model = load_model(model_name)

    original_state = get_model_state(
        original_model
    )

    model_parameters = get_model_parameters(
        original_model
    )

    model_buffers = get_model_buffers(
        original_model
    )

    print(
        f"Original state tensors: "
        f"{len(original_state)}"
    )

    print(
        f"Model parameters: "
        f"{len(model_parameters)}"
    )

    print(
        f"Model buffers: "
        f"{len(model_buffers)}"
    )

    # ------------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------------

    weights = MODEL_CONFIGS[model_name]["weights"]

    print("\nLoading ImageNet validation dataset...")

    dataset = ImageFolder(
        IMAGENET_VAL_DIR,
        transform=weights.transforms(),
    )

    print(
        f"Validation images: "
        f"{len(dataset):,}"
    )

    # ------------------------------------------------------------------------
    # Condition 1
    # ------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("CONDITION 1: ORIGINAL PRETRAINED MODEL")
    print("=" * 80)

    print(
        "Original inference is skipped; "
        "using supplied/reference accuracy."
    )

    result_original = {
        "top1": original_top1,
        "top5": original_top5,
    }

    # ------------------------------------------------------------------------
    # Load StaticBAC reconstruction
    # ------------------------------------------------------------------------

    reconstruction = load_reconstruction(
        reconstruction_path
    )

    # ------------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------------

    check_coverage(
        original_model,
        reconstruction,
    )

    # ------------------------------------------------------------------------
    # Condition 2:
    # reconstructed parameters + original buffers
    # ------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print(
        "CONDITION 2: "
        "RECONSTRUCTED PARAMETERS + ORIGINAL BUFFERS"
    )
    print("=" * 80)

    model_parameters_only = load_model(
        model_name
    )

    load_reconstructed_parameters_only(
        model_parameters_only,
        reconstruction,
    )

    verify_loaded_parameters(
        model_parameters_only,
        reconstruction,
    )

    model_parameters_only.to(DEVICE)

    top1_parameters, top5_parameters = evaluate_imagenet(
        model_parameters_only,
        dataset,
        batch_size,
        args.workers,
    )

    result_parameters = {
        "top1": top1_parameters,
        "top5": top5_parameters,
    }

    # ------------------------------------------------------------------------
    # Condition 3
    #
    # Since buffers are intentionally excluded from StaticBAC inference,
    # condition 3 is no longer a separate condition.
    #
    # Keeping this as a separate inference condition would produce the same
    # model as condition 2.
    # ------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print(
        "BUFFER HANDLING"
    )
    print("=" * 80)

    print(
        "StaticBAC buffer entries found in the NPZ were ignored."
    )

    print(
        "Inference uses the original pretrained model buffers."
    )

    print(
        "Therefore there is no separate reconstructed-buffer "
        "inference condition."
    )

    # ------------------------------------------------------------------------
    # Reconstruction error
    # ------------------------------------------------------------------------

    analyze_reconstruction(
        model_parameters_only,
        original_state,
        reconstruction,
    )

    # ------------------------------------------------------------------------
    # Final comparison
    # ------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print(
        f"\nOriginal pretrained model:"
        f"\n  Top-1: {result_original['top1']:.3f}%"
        f"\n  Top-5: {result_original['top5']:.3f}%"
    )

    print(
        f"\nStaticBAC reconstructed parameters:"
        f"\n  Top-1: {result_parameters['top1']:.3f}%"
        f"\n  Top-5: {result_parameters['top5']:.3f}%"
    )

    print(
        f"\nDifference from original:"
        f"\n  Top-1: "
        f"{result_parameters['top1'] - result_original['top1']:+.3f} pp"
        f"\n  Top-5: "
        f"{result_parameters['top5'] - result_original['top5']:+.3f} pp"
    )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()