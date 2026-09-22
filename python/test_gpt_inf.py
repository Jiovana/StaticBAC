import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import math
import transformers

transformers.logging.set_verbosity_error()


# ======================================================================
# Configuration
# ======================================================================

MODEL_NAME = "openai-community/gpt2"

# StaticBAC reconstructed NPZ
RECONSTRUCTION_PATH = (
    "gpt2_0075_reconstructed.npz"
)

# WikiText-2 evaluation configuration
MAX_LENGTH = 512
STRIDE = 256


# ======================================================================
# Evaluation
# ======================================================================

def perplexity_sliding_window(
    model,
    tokenizer,
    text,
    max_length=512,
    stride=256
):
    """
    Evaluate causal language model perplexity using sliding windows.

    Each token contributes to the loss exactly once.
    The overlapping portion of each window is used only as context.
    """

    model.eval()

    # --------------------------------------------------------------
    # Tokenize entire text
    # --------------------------------------------------------------

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=True
    )

    input_ids = enc["input_ids"][0]
    n_tokens = input_ids.size(0)

    print(f"Total tokens: {n_tokens:,}")

    # --------------------------------------------------------------
    # Sliding-window evaluation
    # --------------------------------------------------------------

    total_loss = 0.0
    total_count = 0

    previous_end = 0

    for begin_idx in tqdm(
        range(0, n_tokens, stride),
        desc="Evaluating"
    ):

        end_idx = min(
            begin_idx + max_length,
            n_tokens
        )

        input_ids_chunk = input_ids[
            begin_idx:end_idx
        ]

        # Number of tokens already evaluated
        # in previous windows.
        already_seen = previous_end - begin_idx

        if already_seen < 0:
            already_seen = 0

        # ----------------------------------------------------------
        # Labels
        # ----------------------------------------------------------

        labels = input_ids_chunk.clone()

        if already_seen > 0:
            labels[:already_seen] = -100

        # ----------------------------------------------------------
        # Batch dimension
        # ----------------------------------------------------------

        input_batch = (
            input_ids_chunk
            .unsqueeze(0)
            .to(model.device)
        )

        labels = (
            labels
            .unsqueeze(0)
            .to(model.device)
        )

        # ----------------------------------------------------------
        # Inference
        # ----------------------------------------------------------

        with torch.no_grad():

            outputs = model(
                input_batch,
                labels=labels
            )

            valid = (
                labels != -100
            ).sum().item()

            if valid == 0:
                continue

            nll = outputs.loss.item() * valid

            total_loss += nll
            total_count += valid

        previous_end = end_idx

        if end_idx == n_tokens:
            break

    # --------------------------------------------------------------
    # Final perplexity
    # --------------------------------------------------------------

    avg_nll = total_loss / total_count
    ppl = math.exp(avg_nll)

    return ppl, avg_nll, total_count


# ======================================================================
# Parse StaticBAC NPZ
# ======================================================================

def read_staticbac_npz(path):
    """
    Read a StaticBAC reconstructed NPZ.

    Expected parameter keys:

        param_000_transformer.wte.weight
        param_001_transformer.wpe.weight
        ...

    Buffer entries, if present, are ignored.
    """

    print(f"Loading StaticBAC reconstruction:")
    print(f"  {path}")

    data = np.load(
        path,
        allow_pickle=False
    )

    parameters = {}
    buffers = {}

    for key in data.files:

        if key.startswith("param_"):

            # ------------------------------------------------------
            # Remove:
            #
            #   param_<id>_
            #
            # Example:
            #
            #   param_000_transformer.wte.weight
            #
            # becomes:
            #
            #   transformer.wte.weight
            # ------------------------------------------------------

            parts = key.split("_", 2)

            if len(parts) != 3:
                raise RuntimeError(
                    f"Invalid StaticBAC parameter key: {key}"
                )

            try:
                tensor_id = int(parts[1])
            except ValueError:
                raise RuntimeError(
                    f"Invalid tensor ID in key: {key}"
                )

            name = parts[2]

            if name in parameters:
                raise RuntimeError(
                    f"Duplicate parameter name in NPZ: {name}"
                )

            parameters[name] = {
                "id": tensor_id,
                "array": data[key]
            }

        elif key.startswith("buffer_"):

            # Buffers are deliberately ignored for the current
            # inference experiment.
            parts = key.split("_", 2)

            if len(parts) == 3:
                try:
                    tensor_id = int(parts[1])
                except ValueError:
                    tensor_id = -1

                name = parts[2]

                buffers[name] = {
                    "id": tensor_id,
                    "array": data[key]
                }

        else:

            print(
                f"WARNING: Ignoring unrecognized NPZ key: {key}"
            )

    print(
        f"Loaded parameter tensors: {len(parameters)}"
    )

    print(
        f"Loaded buffer tensors:    {len(buffers)} "
        f"(ignored)"
    )

    return parameters, buffers


# ======================================================================
# Compare parameter coverage
# ======================================================================

def check_parameter_coverage(
    model,
    reconstructed
):
    """
    Verify that the StaticBAC NPZ contains exactly the model's
    named parameters.

    named_parameters() is used instead of state_dict() because GPT-2
    has tied transformer.wte.weight / lm_head.weight parameters.
    """

    model_parameters = {
        name: tensor
        for name, tensor in model.named_parameters()
    }

    reconstructed_names = set(
        reconstructed.keys()
    )

    model_names = set(
        model_parameters.keys()
    )

    missing = (
        model_names
        - reconstructed_names
    )

    extra = (
        reconstructed_names
        - model_names
    )

    print("\n" + "=" * 70)
    print("RECONSTRUCTION COVERAGE CHECK")
    print("=" * 70)

    print(
        f"Model named parameters:       "
        f"{len(model_names)}"
    )

    print(
        f"Reconstructed parameters:     "
        f"{len(reconstructed_names)}"
    )

    print(
        f"Missing parameters:           "
        f"{len(missing)}"
    )

    print(
        f"Extra reconstructed entries:  "
        f"{len(extra)}"
    )

    if missing:

        print("\nMISSING PARAMETERS:")

        for name in sorted(missing):
            print("  ", name)

        raise RuntimeError(
            f"StaticBAC reconstruction is incomplete: "
            f"{len(missing)} parameters are missing."
        )

    if extra:

        print("\nEXTRA PARAMETERS:")

        for name in sorted(extra):
            print("  ", name)

        raise RuntimeError(
            f"StaticBAC reconstruction contains "
            f"{len(extra)} unexpected parameters."
        )

    print(
        "\nParameter names match exactly."
    )

    return model_parameters


# ======================================================================
# Load StaticBAC parameters into model
# ======================================================================

def load_reconstructed_parameters(
    model,
    reconstructed
):
    """
    Copy reconstructed StaticBAC parameters into the model.

    Model buffers are intentionally left unchanged.
    """

    model_parameters = {
        name: tensor
        for name, tensor in model.named_parameters()
    }

    print("\n" + "=" * 70)
    print("LOADING STATICBAC RECONSTRUCTION")
    print("=" * 70)

    with torch.no_grad():

        for name, entry in reconstructed.items():

            reconstructed_np = entry["array"]

            reconstructed_tensor = torch.from_numpy(
                reconstructed_np
            )

            actual = model_parameters[name]

            # ------------------------------------------------------
            # Shape check
            # ------------------------------------------------------

            if tuple(reconstructed_tensor.shape) != tuple(
                actual.shape
            ):

                raise RuntimeError(
                    f"Shape mismatch for {name}: "
                    f"reconstructed="
                    f"{tuple(reconstructed_tensor.shape)}, "
                    f"model="
                    f"{tuple(actual.shape)}"
                )

            # ------------------------------------------------------
            # Dtype check / conversion
            #
            # StaticBAC reconstruction is normally float32.
            # Convert explicitly to the model parameter dtype.
            # ------------------------------------------------------

            reconstructed_tensor = (
                reconstructed_tensor.to(
                    dtype=actual.dtype
                )
            )

            # ------------------------------------------------------
            # Copy
            # ------------------------------------------------------

            actual.copy_(
                reconstructed_tensor
            )

    print(
        "All reconstructed parameters copied into model."
    )


# ======================================================================
# Verify reconstruction load
# ======================================================================

def verify_reconstruction_load(
    model,
    reconstructed
):
    """
    Verify that every reconstructed parameter was copied exactly.
    """

    model_parameters = {
        name: tensor
        for name, tensor in model.named_parameters()
    }

    print("\n" + "=" * 70)
    print("VERIFYING RECONSTRUCTION LOAD")
    print("=" * 70)

    failures = []

    for name, entry in reconstructed.items():

        expected = torch.from_numpy(
            entry["array"]
        ).to(
            dtype=model_parameters[name].dtype,
            device=model_parameters[name].device
        )

        actual = model_parameters[name]

        if not torch.equal(
            actual,
            expected
        ):
            failures.append(name)

    if failures:

        print(
            "\nRECONSTRUCTION VERIFICATION FAILED:"
        )

        for name in failures:
            print("  ", name)

        raise RuntimeError(
            f"{len(failures)} parameters "
            f"did not match after loading."
        )

    print(
        "All reconstructed parameters loaded and "
        "verified successfully."
    )


# ======================================================================
# Compare original vs reconstructed
# ======================================================================

def compare_original_reconstructed(
    original_state,
    model
):
    """
    Compare original pretrained parameters against the reconstructed
    parameters currently loaded into the model.

    Buffers are deliberately excluded.
    """

    model_parameters = {
        name: tensor
        for name, tensor in model.named_parameters()
    }

    print("\n" + "=" * 70)
    print("ORIGINAL vs RECONSTRUCTED PARAMETER ANALYSIS")
    print("=" * 70)

    num_different = 0

    total_elements = 0
    different_elements = 0

    global_max_abs_diff = 0.0
    global_sum_abs_diff = 0.0

    tensor_statistics = []

    for name in model_parameters.keys():

        original = (
            original_state[name]
            .float()
        )

        reconstructed = (
            model_parameters[name]
            .float()
        )

        diff = (
            original
            - reconstructed
        ).abs()

        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        num_elements = diff.numel()

        num_diff_elements = (
            torch.count_nonzero(diff)
            .item()
        )

        total_elements += num_elements

        different_elements += (
            num_diff_elements
        )

        global_max_abs_diff = max(
            global_max_abs_diff,
            max_diff
        )

        global_sum_abs_diff += (
            diff.sum().item()
        )

        if max_diff > 0:
            num_different += 1

        tensor_statistics.append(
            (
                name,
                max_diff,
                mean_diff,
                num_diff_elements,
                num_elements
            )
        )

    # --------------------------------------------------------------
    # Global statistics
    # --------------------------------------------------------------

    global_mean_abs_diff = (
        global_sum_abs_diff
        / total_elements
    )

    percent_elements_changed = (
        100.0
        * different_elements
        / total_elements
    )

    print(
        f"\nParameters changed: "
        f"{num_different}/"
        f"{len(model_parameters)}"
    )

    print(
        f"Elements changed: "
        f"{different_elements:,}/"
        f"{total_elements:,} "
        f"({percent_elements_changed:.4f}%)"
    )

    print(
        f"Global mean absolute difference: "
        f"{global_mean_abs_diff:.8g}"
    )

    print(
        f"Global maximum absolute difference: "
        f"{global_max_abs_diff:.8g}"
    )

    # --------------------------------------------------------------
    # Largest differences
    # --------------------------------------------------------------

    print(
        "\nLargest parameter differences:"
    )

    tensor_statistics.sort(
        key=lambda x: x[1],
        reverse=True
    )

    for (
        name,
        max_diff,
        mean_diff,
        num_diff_elements,
        num_elements
    ) in tensor_statistics[:20]:

        percent_changed = (
            100.0
            * num_diff_elements
            / num_elements
        )

        print(
            f"\n{name}"
            f"\n  max |diff|:      "
            f"{max_diff:.8g}"
            f"\n  mean |diff|:     "
            f"{mean_diff:.8g}"
            f"\n  changed:         "
            f"{num_diff_elements:,}/"
            f"{num_elements:,} "
            f"({percent_changed:.4f}%)"
        )


# ======================================================================
# Inspect representative tensor
# ======================================================================

def inspect_embedding(
    original_state,
    model
):
    """
    Print a representative GPT-2 embedding tensor.
    """

    model_parameters = {
        name: tensor
        for name, tensor in model.named_parameters()
    }

    embedding_candidates = [
        name
        for name in model_parameters.keys()
        if (
            "wte" in name.lower()
            or "embedding" in name.lower()
        )
    ]

    if not embedding_candidates:
        return

    sample_name = embedding_candidates[0]

    print("\n" + "=" * 70)
    print("SAMPLE TENSOR")
    print("=" * 70)

    original = original_state[
        sample_name
    ]

    reconstructed = model_parameters[
        sample_name
    ]

    print(
        f"Tensor: {sample_name}"
    )

    print(
        f"Shape:  {tuple(original.shape)}"
    )

    print(
        f"Original dtype:      "
        f"{original.dtype}"
    )

    print(
        f"Reconstructed dtype: "
        f"{reconstructed.dtype}"
    )

    print("\nFirst 10 original values:")
    print(
        original.view(-1)[:10]
    )

    print("\nFirst 10 reconstructed values:")
    print(
        reconstructed.view(-1)[:10]
    )

    print("\nFirst 10 absolute differences:")
    print(
        (
            original.float()
            - reconstructed.float()
        )
        .abs()
        .view(-1)[:10]
    )


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":

    # ==================================================================
    # LOAD TOKENIZER AND DATASET
    # ==================================================================

    print("\n" + "=" * 70)
    print("LOADING TOKENIZER AND DATASET")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="validation"
    )

    # Remove empty lines.

    text = "\n".join(
        [
            t
            for t in dataset["text"]
            if t.strip()
        ]
    )

    print(
        f"WikiText-2 validation entries: "
        f"{len(dataset)}"
    )


    # ==================================================================
    # LOAD ORIGINAL MODEL
    # ==================================================================

    print("\n" + "=" * 70)
    print("LOADING ORIGINAL MODEL")
    print("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    ).to("cpu")

    model.eval()

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        "Embedding shape:",
        model.transformer.wte.weight.shape
    )

    print(
        "LM head shape:",
        model.lm_head.weight.shape
    )

    print(
        "Same underlying parameter:",
        model.transformer.wte.weight.data_ptr()
        == model.lm_head.weight.data_ptr()
    )

    print(
        "Same values:",
        torch.equal(
            model.transformer.wte.weight,
            model.lm_head.weight
        )
    )


    # ==================================================================
    # SAVE ORIGINAL PARAMETERS
    # ==================================================================

    # Only named parameters are saved here.
    #
    # This intentionally follows StaticBAC's parameter representation.
    # Buffers are not included because the current experiment uses
    # original pretrained buffers.

    original_state = {
        name: tensor.clone()
        for name, tensor in model.named_parameters()
    }

    print(
        f"Original model parameters: "
        f"{len(original_state)}"
    )


    # ==================================================================
    # BASELINE INFERENCE
    # ==================================================================

    print("\n" + "=" * 70)
    print("BASELINE INFERENCE")
    print("=" * 70)

    baseline_ppl, baseline_nll, baseline_tokens = (
        perplexity_sliding_window(
            model,
            tokenizer,
            text,
            max_length=MAX_LENGTH,
            stride=STRIDE
        )
    )

    print(
        f"\nBaseline perplexity: "
        f"{baseline_ppl:.8f}"
    )

    print(
        f"Baseline avg NLL:     "
        f"{baseline_nll:.8f}"
    )

    print(
        f"Baseline tokens:      "
        f"{baseline_tokens:,}"
    )


    # ==================================================================
    # LOAD STATICBAC RECONSTRUCTION
    # ==================================================================

    print("\n" + "=" * 70)
    print("LOADING STATICBAC RECONSTRUCTION")
    print("=" * 70)

    reconstructed, ignored_buffers = (
        read_staticbac_npz(
            RECONSTRUCTION_PATH
        )
    )


    # ==================================================================
    # CHECK PARAMETER COVERAGE
    # ==================================================================

    check_parameter_coverage(
        model,
        reconstructed
    )


    # ==================================================================
    # LOAD RECONSTRUCTED PARAMETERS
    # ==================================================================

    load_reconstructed_parameters(
        model,
        reconstructed
    )


    # ==================================================================
    # VERIFY RECONSTRUCTION LOAD
    # ==================================================================

    verify_reconstruction_load(
        model,
        reconstructed
    )


    # ==================================================================
    # VERIFY GPT-2 WEIGHT TYING
    # ==================================================================

    print("\n" + "=" * 70)
    print("VERIFYING GPT-2 WEIGHT TYING")
    print("=" * 70)

    tied = (
        model.transformer.wte.weight.data_ptr()
        == model.lm_head.weight.data_ptr()
    )

    same_values = torch.equal(
        model.transformer.wte.weight,
        model.lm_head.weight
    )

    print(
        "Embedding / LM head share storage:",
        tied
    )

    print(
        "Embedding / LM head values identical:",
        same_values
    )

    if not tied:

        raise RuntimeError(
            "GPT-2 embedding and LM head are no longer tied."
        )

    if not same_values:

        raise RuntimeError(
            "GPT-2 embedding and LM head contain different values."
        )

    print(
        "GPT-2 tied weights verified."
    )


    # ==================================================================
    # COMPARE ORIGINAL vs RECONSTRUCTED
    # ==================================================================

    compare_original_reconstructed(
        original_state,
        model
    )


    # ==================================================================
    # SAMPLE TENSOR
    # ==================================================================

    inspect_embedding(
        original_state,
        model
    )


    # ==================================================================
    # RECONSTRUCTED MODEL INFERENCE
    # ==================================================================

    print("\n" + "=" * 70)
    print(
        "RECONSTRUCTED MODEL INFERENCE"
    )
    print("=" * 70)

    reconstructed_ppl, reconstructed_nll, reconstructed_tokens = (
        perplexity_sliding_window(
            model,
            tokenizer,
            text,
            max_length=MAX_LENGTH,
            stride=STRIDE
        )
    )

    print(
        f"\nReconstructed perplexity: "
        f"{reconstructed_ppl:.8f}"
    )

    print(
        f"Reconstructed avg NLL:     "
        f"{reconstructed_nll:.8f}"
    )

    print(
        f"Reconstructed tokens:      "
        f"{reconstructed_tokens:,}"
    )


    # ==================================================================
    # FINAL COMPARISON
    # ==================================================================

    ppl_difference = (
        reconstructed_ppl
        - baseline_ppl
    )

    nll_difference = (
        reconstructed_nll
        - baseline_nll
    )

    relative_ppl_change = (
        100.0
        * ppl_difference
        / baseline_ppl
    )

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        f"Baseline perplexity:       "
        f"{baseline_ppl:.8f}"
    )

    print(
        f"Reconstructed perplexity:  "
        f"{reconstructed_ppl:.8f}"
    )

    print(
        f"Perplexity difference:     "
        f"{ppl_difference:+.8f}"
    )

    print(
        f"Perplexity difference (%): "
        f"{relative_ppl_change:+.4f}%"
    )

    print(
        f"\nBaseline avg NLL:          "
        f"{baseline_nll:.8f}"
    )

    print(
        f"Reconstructed avg NLL:     "
        f"{reconstructed_nll:.8f}"
    )

    print(
        f"NLL difference:            "
        f"{nll_difference:+.8f}"
    )

    if abs(ppl_difference) < 1e-12:

        print(
            "\nNOTE: Baseline and reconstructed "
            "models have identical perplexity."
        )

    elif ppl_difference > 0:

        print(
            "\nNOTE: Reconstruction increased "
            "perplexity."
        )

    else:

        print(
            "\nNOTE: Reconstruction decreased "
            "perplexity."
        )

    print("\nDone.")