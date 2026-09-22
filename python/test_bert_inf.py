import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset
from tqdm import tqdm
import transformers

transformers.logging.set_verbosity_error()


# ======================================================================
# Configuration
# ======================================================================

MODEL_NAME = "google-bert/bert-base-uncased"

# StaticBAC reconstructed NPZ
RECONSTRUCTION_PATH = (
    "bert_015_reconstructed.npz"
)

# WikiText-2 evaluation configuration
MAX_LENGTH = 512
# BERT MLM configuration
MASK_PROBABILITY = 0.15

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ======================================================================
# Load WikiText-2
# ======================================================================

def load_wikitext():

    print("\n" + "=" * 70)
    print("LOADING WIKITEXT-2")
    print("=" * 70)

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="validation"
    )

    texts = [
        text
        for text in dataset["text"]
        if text.strip()
    ]

    text = "\n".join(texts)

    print(
        f"WikiText-2 validation entries: "
        f"{len(dataset)}"
    )

    print(
        f"Non-empty entries used: "
        f"{len(texts)}"
    )

    print(
        f"Total characters: "
        f"{len(text):,}"
    )

    return text


# ======================================================================
# Tokenize WikiText-2
# ======================================================================

def tokenize_wikitext(
    tokenizer,
    text
):

    print("\n" + "=" * 70)
    print("TOKENIZING WIKITEXT-2")
    print("=" * 70)

    # --------------------------------------------------------------
    # Tokenize without truncation.
    # --------------------------------------------------------------

    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False
    )["input_ids"]

    print(
        f"Total content tokens: "
        f"{len(token_ids):,}"
    )

    # --------------------------------------------------------------
    # BERT sequences:
    #
    # [CLS] + 510 content tokens + [SEP]
    #
    # This gives a maximum sequence length of 512.
    # --------------------------------------------------------------

    chunk_size = MAX_LENGTH - 2

    sequences = []

    for start in range(
        0,
        len(token_ids),
        chunk_size
    ):

        chunk = token_ids[
            start:start + chunk_size
        ]

        if len(chunk) < 2:
            continue

        sequence = (
            [tokenizer.cls_token_id]
            + chunk
            + [tokenizer.sep_token_id]
        )

        # ----------------------------------------------------------
        # Pad shorter final sequence.
        # ----------------------------------------------------------

        padding_length = (
            MAX_LENGTH
            - len(sequence)
        )

        sequence += (
            [tokenizer.pad_token_id]
            * padding_length
        )

        sequences.append(sequence)

    input_ids = torch.tensor(
        sequences,
        dtype=torch.long
    )

    attention_mask = (
        input_ids != tokenizer.pad_token_id
    ).long()

    encodings = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

    print(
        f"BERT input shape: "
        f"{tuple(input_ids.shape)}"
    )

    print(
        f"Number of sequences: "
        f"{input_ids.shape[0]:,}"
    )

    return encodings


# ======================================================================
# Create MLM masks
# ======================================================================

def create_mlm_batch(
    encodings,
    tokenizer
):

    print("\n" + "=" * 70)
    print("CREATING MLM MASKS")
    print("=" * 70)

    input_ids = encodings["input_ids"].clone()
    attention_mask = encodings["attention_mask"]

    labels = input_ids.clone()

    # --------------------------------------------------------------
    # Random generator
    #
    # No global seed is required.
    #
    # The same mlm_inputs object is used for both the original
    # and reconstructed models, so both models see exactly the
    # same masked inputs.
    # --------------------------------------------------------------

    rng = np.random.default_rng(SEED)

    # --------------------------------------------------------------
    # Identify special tokens
    # --------------------------------------------------------------

    special_masks = []

    for ids in input_ids:

        special_masks.append(
            tokenizer.get_special_tokens_mask(
                ids.tolist(),
                already_has_special_tokens=True
            )
        )

    special_masks = torch.tensor(
        special_masks,
        dtype=torch.bool
    )

    # --------------------------------------------------------------
    # Select tokens for prediction
    # --------------------------------------------------------------

    random_values = torch.from_numpy(
        rng.random(input_ids.shape)
    )

    candidate_mask = (
        attention_mask.bool()
        & ~special_masks
    )

    masked_positions = (
        candidate_mask
        & (random_values < MASK_PROBABILITY)
    )

    labels[~masked_positions] = -100

    # --------------------------------------------------------------
    # 80% [MASK]
    # 10% random token
    # 10% unchanged
    # --------------------------------------------------------------

    mask_random = rng.random(
        input_ids.shape
    )

    mask_positions = (
        masked_positions
        & torch.from_numpy(
            mask_random < 0.80
        )
    )

    random_positions = (
        masked_positions
        & torch.from_numpy(
            (mask_random >= 0.80)
            & (mask_random < 0.90)
        )
    )

    input_ids[mask_positions] = (
        tokenizer.mask_token_id
    )

    random_token_ids = torch.from_numpy(
        rng.integers(
            low=0,
            high=tokenizer.vocab_size,
            size=input_ids.shape
        )
    ).long()

    input_ids[random_positions] = (
        random_token_ids[random_positions]
    )

    masked_tokens = (
        masked_positions.sum().item()
    )

    print(
        f"Masked tokens: "
        f"{masked_tokens:,}"
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# ======================================================================
# MLM evaluation
# ======================================================================

def evaluate_mlm(
    model,
    inputs,
    batch_size=8
):

    model.eval()

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    labels = inputs["labels"]

    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    num_samples = input_ids.shape[0]

    print(
        f"Evaluating {num_samples} samples "
        f"with batch size {batch_size}..."
    )

    with torch.no_grad():

        for start in tqdm(
            range(0, num_samples, batch_size),
            desc="MLM inference"
        ):

            end = min(
                start + batch_size,
                num_samples
            )

            batch_input_ids = (
                input_ids[start:end]
                .to(DEVICE)
            )

            batch_attention_mask = (
                attention_mask[start:end]
                .to(DEVICE)
            )

            batch_labels = (
                labels[start:end]
                .to(DEVICE)
            )

            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_labels
            )

            logits = outputs.logits

            masked = (
                batch_labels != -100
            )

            masked_count = (
                masked.sum().item()
            )

            total_loss += (
                outputs.loss.item()
                * masked_count
            )

            predictions = logits.argmax(
                dim=-1
            )

            total_correct += (
                (
                    predictions == batch_labels
                )
                & masked
            ).sum().item()

            total_masked += masked_count

    mean_loss = (
        total_loss / total_masked
    )

    accuracy = (
        total_correct / total_masked
    )

    return {
        "loss": mean_loss,
        "accuracy": accuracy,
        "masked_tokens": total_masked
    }


# ======================================================================
# Parse StaticBAC NPZ
# ======================================================================

def read_staticbac_npz(path):

    print("\n" + "=" * 70)
    print("LOADING STATICBAC RECONSTRUCTION")
    print("=" * 70)

    print(f"Path: {path}")

    data = np.load(
        path,
        allow_pickle=False
    )

    parameters = {}
    buffers = {}

    for key in data.files:

        if key.startswith("param_"):

            parts = key.split("_", 2)

            if len(parts) != 3:
                raise RuntimeError(
                    f"Invalid parameter key: {key}"
                )

            tensor_id = int(parts[1])
            tensor_name = parts[2]

            parameters[tensor_name] = {
                "id": tensor_id,
                "array": data[key]
            }

        elif key.startswith("buffer_"):

            parts = key.split("_", 2)

            if len(parts) != 3:
                raise RuntimeError(
                    f"Invalid buffer key: {key}"
                )

            tensor_id = int(parts[1])
            tensor_name = parts[2]

            buffers[tensor_name] = {
                "id": tensor_id,
                "array": data[key]
            }

        else:

            print(
                f"WARNING: Ignoring NPZ entry: {key}"
            )

    print(
        f"Parameter tensors: {len(parameters)}"
    )

    print(
        f"Buffer tensors:    {len(buffers)} "
        f"(ignored)"
    )

    return parameters, buffers


# ======================================================================
# Parameter coverage
# ======================================================================

def check_parameter_coverage(
    model,
    reconstructed
):

    model_parameters = {
        name: tensor
        for name, tensor in model.named_parameters()
    }

    model_names = set(
        model_parameters.keys()
    )

    reconstructed_names = set(
        reconstructed.keys()
    )

    matching = sorted(
        model_names & reconstructed_names
    )

    missing = sorted(
        model_names - reconstructed_names
    )

    extra = sorted(
        reconstructed_names - model_names
    )

    print("\n" + "=" * 70)
    print("RECONSTRUCTION COVERAGE CHECK")
    print("=" * 70)

    print(
        f"Model parameters:       "
        f"{len(model_names)}"
    )

    print(
        f"Reconstructed entries:  "
        f"{len(reconstructed_names)}"
    )

    print(
        f"Matching parameters:    "
        f"{len(matching)}"
    )

    print(
        f"Missing parameters:     "
        f"{len(missing)}"
    )

    print(
        f"Extra NPZ entries:      "
        f"{len(extra)}"
    )

    if missing:

        print("\nMissing parameters:")

        for name in missing:
            print(f"  {name}")

        raise RuntimeError(
            "StaticBAC reconstruction is incomplete."
        )

    if extra:

        print("\nExtra NPZ entries:")

        for name in extra:
            print(f"  {name}")

        raise RuntimeError(
            "StaticBAC reconstruction contains "
            "unexpected entries."
        )

    print(
        "\nParameter names match exactly."
    )

    return matching


# ======================================================================
# Load reconstructed parameters
# ======================================================================

def load_reconstructed_parameters(
    model,
    reconstructed,
    matching
):

    print("\n" + "=" * 70)
    print("LOADING STATICBAC PARAMETERS")
    print("=" * 70)

    model_parameters = {
        name: tensor
        for name, tensor in model.named_parameters()
    }

    with torch.no_grad():

        for name in matching:

            model_tensor = (
                model_parameters[name]
            )

            reconstructed_tensor = (
                torch.from_numpy(
                    reconstructed[name]["array"]
                )
            )

            if tuple(
                model_tensor.shape
            ) != tuple(
                reconstructed_tensor.shape
            ):

                raise RuntimeError(
                    f"Shape mismatch for {name}: "
                    f"model="
                    f"{tuple(model_tensor.shape)}, "
                    f"NPZ="
                    f"{tuple(reconstructed_tensor.shape)}"
                )

            reconstructed_tensor = (
                reconstructed_tensor.to(
                    device=model_tensor.device,
                    dtype=model_tensor.dtype
                )
            )

            model_tensor.copy_(
                reconstructed_tensor
            )

    print(
        "All reconstructed parameters loaded."
    )


# ======================================================================
# Verify reconstruction
# ======================================================================

def verify_reconstruction_load(
    model,
    reconstructed,
    matching
):

    print("\n" + "=" * 70)
    print("VERIFYING RECONSTRUCTION")
    print("=" * 70)

    model_parameters = {
        name: tensor
        for name, tensor in model.named_parameters()
    }

    max_error = 0.0

    for name in matching:

        model_tensor = (
            model_parameters[name]
            .detach()
            .cpu()
            .numpy()
        )

        reconstructed_tensor = (
            reconstructed[name]["array"]
        )

        error = np.max(
            np.abs(
                model_tensor.astype(np.float64)
                - reconstructed_tensor.astype(np.float64)
            )
        )

        max_error = max(
            max_error,
            float(error)
        )

    print(
        f"Maximum difference: "
        f"{max_error:.10g}"
    )

    if max_error == 0:

        print(
            "Exact reconstruction load verified."
        )

    else:

        print(
            "WARNING: Non-zero difference detected."
        )


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("BERT STATICBAC MACHINE-FIDELITY EVALUATION")
    print("=" * 70)

    print(
        f"Model:        {MODEL_NAME}"
    )

    print(
        f"NPZ:          {RECONSTRUCTION_PATH}"
    )

    print(
        f"Device:       {DEVICE}"
    )

    print(
        f"Max length:   {MAX_LENGTH}"
    )

    print(
        f"Mask prob.:   {MASK_PROBABILITY}"
    )


    # ==================================================================
    # LOAD WIKITEXT-2 FIRST
    # ==================================================================

    text = load_wikitext()


    # ==================================================================
    # LOAD TOKENIZER
    # ==================================================================

    print("\n" + "=" * 70)
    print("LOADING TOKENIZER")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print(
        f"Tokenizer: {MODEL_NAME}"
    )


    # ==================================================================
    # TOKENIZE
    # ==================================================================

    encodings = tokenize_wikitext(
        tokenizer,
        text
    )


    # ==================================================================
    # CREATE FIXED MLM INPUT
    # ==================================================================

    mlm_inputs = create_mlm_batch(
        encodings,
        tokenizer
    )


    # ==================================================================
    # LOAD ORIGINAL MODEL
    # ==================================================================

    print("\n" + "=" * 70)
    print("LOADING ORIGINAL MODEL")
    print("=" * 70)

    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME
    ).to(DEVICE)

    model.eval()

    print(
        f"Model: {MODEL_NAME}"
    )


    # ==================================================================
    # ORIGINAL INFERENCE
    # ==================================================================

    print("\n" + "=" * 70)
    print("ORIGINAL MODEL INFERENCE")
    print("=" * 70)

    original_results = evaluate_mlm(
        model,
        mlm_inputs,
        batch_size=8
    )

    print("\nOriginal BERT:")

    print(
        f"  MLM loss:     "
        f"{original_results['loss']:.6f}"
    )

    print(
        f"  MLM accuracy: "
        f"{original_results['accuracy']:.6%}"
    )

    print(
        f"  Masked tokens: "
        f"{original_results['masked_tokens']:,}"
    )


    # ==================================================================
    # LOAD STATICBAC RECONSTRUCTION
    # ==================================================================

    reconstructed, ignored_buffers = (
        read_staticbac_npz(
            RECONSTRUCTION_PATH
        )
    )

    matching = check_parameter_coverage(
        model,
        reconstructed
    )


    # ==================================================================
    # LOAD RECONSTRUCTED PARAMETERS
    # ==================================================================

    load_reconstructed_parameters(
        model,
        reconstructed,
        matching
    )


    # ==================================================================
    # VERIFY RECONSTRUCTION
    # ==================================================================

    verify_reconstruction_load(
        model,
        reconstructed,
        matching
    )


    # ==================================================================
    # RECONSTRUCTED INFERENCE
    # ==================================================================

    print("\n" + "=" * 70)
    print("STATICBAC RECONSTRUCTED MODEL INFERENCE")
    print("=" * 70)

    reconstructed_results = evaluate_mlm(
        model,
        mlm_inputs,
        batch_size=8
    )

    print("\nStaticBAC reconstructed BERT:")

    print(
        f"  MLM loss:     "
        f"{reconstructed_results['loss']:.6f}"
    )

    print(
        f"  MLM accuracy: "
        f"{reconstructed_results['accuracy']:.6%}"
    )

    print(
        f"  Masked tokens: "
        f"{reconstructed_results['masked_tokens']:,}"
    )


    # ==================================================================
    # FINAL COMPARISON
    # ==================================================================

    loss_difference = (
        reconstructed_results["loss"]
        - original_results["loss"]
    )

    accuracy_difference = (
        reconstructed_results["accuracy"]
        - original_results["accuracy"]
    )

    relative_loss_change = (
        100.0
        * loss_difference
        / original_results["loss"]
    )

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        f"Baseline MLM loss:       "
        f"{original_results['loss']:.8f}"
    )

    print(
        f"Reconstructed MLM loss:  "
        f"{reconstructed_results['loss']:.8f}"
    )

    print(
        f"MLM loss difference:     "
        f"{loss_difference:+.8f}"
    )

    print(
        f"MLM loss relative change:"
        f" {relative_loss_change:+.4f}%"
    )

    print()

    print(
        f"Baseline MLM accuracy:       "
        f"{original_results['accuracy']:.6%}"
    )

    print(
        f"Reconstructed MLM accuracy:  "
        f"{reconstructed_results['accuracy']:.6%}"
    )

    print(
        f"MLM accuracy difference:     "
        f"{accuracy_difference:+.6%}"
    )

    print("\nDone.")