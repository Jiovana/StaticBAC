import os
import glob
import re
import pandas as pd


# ============================================================
# Configuration
# ============================================================

CSV_DIR = "predictor_analysis"

OUTPUT_GLOBAL = "predictor_global_by_type.csv"
OUTPUT_MODEL = "predictor_by_model_and_type.csv"
OUTPUT_DETAIL = "predictor_tensor_classification.csv"


# ============================================================
# Tensor classification
# ============================================================

def classify_tensor(row):
    model = str(row["model"]).lower()
    name = str(row["tensor_name"]).lower()
    tensor_type = str(row["tensor_type"]).lower()

    # ========================================================
    # Buffers
    # ========================================================
    if tensor_type == "buffer":
        return "buffer"

    # ========================================================
    # Biases
    # ========================================================
    if tensor_type == "bias" or name.endswith("_bias"):
        return "bias"

    # ========================================================
    # Only weights below
    # ========================================================
    if not (tensor_type == "weight" or name.endswith("_weight")):
        return "other"

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    if any(x in name for x in [
        "embedding",
        "embeddings",
        "token_embedding",
        "position_embedding",
    ]):
        return "embedding_weight"

    # GPT / GPT-2
    if model in ["gpt", "gpt2"]:
        if name.startswith("transformer_wte"):
            return "embedding_weight"
        if name.startswith("transformer_wpe"):
            return "embedding_weight"

    # ========================================================
    # NORMALIZATION
    # ========================================================

    if any(x in name for x in [
        "layer_norm",
        "layernorm",
        "_ln_",
        "ln_",
        "norm_weight",
        "batch_norm",
        "batchnorm",
    ]):
        return "norm_weight"

    # ========================================================
    # VGG19
    # ========================================================

    if model == "vgg19":

        # features_* are convolutional layers
        if name.startswith("features_"):
            return "conv_weight"

        # classifier_* are fully-connected layers
        if name.startswith("classifier_"):
            return "fc_weight"

    # ========================================================
    # RESNET50
    # ========================================================

    if model == "resnet50":

        if (
            name.startswith("bn")
            or "_bn" in name
            or name.endswith("_bn_weight") 
            or "_downsample_1_weight" in name
        ):
            return "norm_weight"

        if "_downsample_0_weight" in name:
            return "conv_weight"

        if (
            name.startswith("conv")
            or "_conv" in name
        ):
            return "conv_weight"

        if name.startswith("fc_"):
            return "fc_weight"

        

    # ========================================================
    # INCEPTIONV3
    # ========================================================

    if model == "inceptionv3":

        if "_bn_" in name or name.endswith("_bn_weight"):
            return "norm_weight"
        if "conv" in name:
            return "conv_weight"

        if name.startswith("fc_"):
            return "fc_weight"

        if name.startswith("auxlogits_"):
            return "fc_weight"

    # ========================================================
    # MOBILENET V3
    # ========================================================

    if model == "mobilenet_v3":

        if name.startswith("classifier_"):
            return "fc_weight"

        # Squeeze-and-excitation layers
        if "_fc1_" in name or "_fc2_" in name:
            return "mlp_weight"

        # Everything else in features is convolutional
        if name.startswith("features_"):
            return "conv_weight"

    # ========================================================
    # EFFICIENTNET B7
    # ========================================================

    if model == "efficientnet_b7":

        if name.startswith("classifier_"):
            return "fc_weight"

        # Squeeze-and-excitation layers
        if "_fc1_" in name or "_fc2_" in name:
            return "mlp_weight"

        # EfficientNet feature extractor
        if name.startswith("features_"):
            return "conv_weight"

    # ========================================================
    # VIT B16
    # ========================================================

    if model == "vitb16":

        if "conv_proj" in name:
            return "conv_weight"

        if any(x in name for x in [
            "self_attention",
            "selfattention",
            "attention",
        ]):
            return "attention_weight"

        if "mlp" in name:
            return "mlp_weight"

        if "heads" in name:
            return "fc_weight"

    # ========================================================
    # BERT
    # ========================================================

    if model == "google-bert":

        if "embedding" in name:
            return "embedding_weight"

        if "output_dense" in name:
            return "mlp_weight"

        if "cls_predictions_transform" in name:
            return "fc_weight"

        if "attention" in name:
            return "attention_weight"

        if "intermediate" in name:
            return "mlp_weight"

        # BERT output sublayer is feed-forward, except
        # attention.output, already caught above.
        if ".output." in name:
            return "mlp_weight"

        if "pooler" in name or "classifier" in name:
            return "fc_weight"

    # ========================================================
    # GPT
    # ========================================================

    if model == "gpt":

        if "positions_embed" in name or "tokens_embed" in name:
            return "embedding_weight"

        if "attn" in name:
            return "attention_weight"

        if "mlp" in name:
            return "mlp_weight"

    # ========================================================
    # GPT-2
    # ========================================================

    if model == "gpt2":

        if "attn" in name:
            return "attention_weight"

        if "mlp" in name:
            return "mlp_weight"

    # ========================================================
    # T5
    # ========================================================

    if model == "t5":

        if name == "transformer_shared_weight":
            return "embedding_weight"

        if any(x in name for x in [
            "selfattention",
            "encdecattention",
            "attention",
        ]):
            return "attention_weight"

        if any(x in name for x in [
            "densereludense",
            "dense_relu_dense",
        ]):
            return "mlp_weight"

        if "classification_head" in name:
            return "fc_weight"

    # ========================================================
    # GENERIC TRANSFORMER FALLBACKS
    # ========================================================

    if any(x in name for x in [
        "selfattention",
        "self_attention",
        "crossattention",
        "cross_attention",
        "encdecattention",
        "enc_dec_attention",
        "attention",
        "q_proj",
        "k_proj",
        "v_proj",
        "qkv",
        "in_proj",
        "out_proj",
    ]):
        return "attention_weight"

    if any(x in name for x in [
        "mlp",
        "feedforward",
        "feed_forward",
        "fc1",
        "fc2",
        "wi_",
        "wo_",
    ]):
        return "mlp_weight"

    # ========================================================
    # Generic convolution fallback
    # ========================================================

    if "conv" in name:
        return "conv_weight"

    # ========================================================
    # Generic fully-connected fallback
    # ========================================================

    if (
        name.startswith("fc_")
        or name.startswith("classifier_")
    ):
        return "fc_weight"

    # ========================================================
    # Unknown weight
    # ========================================================

    return "other_weight"


# ============================================================
# Load all CSVs
# ============================================================

csv_files = sorted(
    glob.glob(
        os.path.join(CSV_DIR, "*.csv")
    )
)

if not csv_files:
    raise RuntimeError(
        f"No CSV files found in: {CSV_DIR}"
    )

print(f"Found {len(csv_files)} CSV files")

all_data = []

for path in csv_files:

    model = os.path.splitext(
        os.path.basename(path)
    )[0]

    df = pd.read_csv(path)

    df["model"] = model

    # Classify tensor
    df["classified_type"] = df.apply(
        classify_tensor,
        axis=1
    )

    all_data.append(df)


data = pd.concat(
    all_data,
    ignore_index=True
)


# ============================================================
# Sanity check
# ============================================================

required_columns = [
    "tensor_name",
    "tensor_type",
    "chunks",
    "used_chunks",
    "skipped_chunks",
    "pred_none",
    "pred_mean",
    "pred_neighbor",
]

missing = [
    c for c in required_columns
    if c not in data.columns
]

if missing:
    raise RuntimeError(
        "Missing required columns: "
        + ", ".join(missing)
    )


# ============================================================
# Global aggregation
# ============================================================

rows = []

for tensor_type, group in data.groupby(
    "classified_type"
):

    total_chunks = group["chunks"].sum()
    used_chunks = group["used_chunks"].sum()
    skipped_chunks = group["skipped_chunks"].sum()

    none = group["pred_none"].sum()
    mean = group["pred_mean"].sum()
    neighbor = group["pred_neighbor"].sum()

    rows.append({
        "tensor_category": tensor_type,

        "tensors": len(group),

        "chunks": total_chunks,
        "used_chunks": used_chunks,
        "skipped_chunks": skipped_chunks,

        "pred_none": none,
        "pred_mean": mean,
        "pred_neighbor": neighbor,

        "pred_none_pct": (
            none / used_chunks
            if used_chunks else 0
        ),

        "pred_mean_pct": (
            mean / used_chunks
            if used_chunks else 0
        ),

        "pred_neighbor_pct": (
            neighbor / used_chunks
            if used_chunks else 0
        ),

        "skip_pct": (
            skipped_chunks / total_chunks
            if total_chunks else 0
        ),
    })


global_df = pd.DataFrame(rows)


# ============================================================
# Add contribution of each category to all used chunks
# ============================================================

total_used_chunks = global_df[
    "used_chunks"
].sum()

global_df["used_chunk_share_pct"] = (
    global_df["used_chunks"]
    / total_used_chunks
    if total_used_chunks
    else 0
)


# ============================================================
# Order categories
# ============================================================

category_order = [
    "conv_weight",
    "attention_weight",
    "mlp_weight",
    "fc_weight",
    "embedding_weight",
    "norm_weight",
    "bias",
    "buffer",
    "other_weight",
    "other",
]

global_df["order"] = global_df[
    "tensor_category"
].apply(
    lambda x:
        category_order.index(x)
        if x in category_order
        else 999
)

global_df = (
    global_df
    .sort_values("order")
    .drop(columns="order")
)


# ============================================================
# Per-model + tensor-category aggregation
# ============================================================

rows = []

for (
    model,
    tensor_type
), group in data.groupby(
    ["model", "classified_type"]
):

    total_chunks = group["chunks"].sum()
    used_chunks = group["used_chunks"].sum()
    skipped_chunks = group["skipped_chunks"].sum()

    none = group["pred_none"].sum()
    mean = group["pred_mean"].sum()
    neighbor = group["pred_neighbor"].sum()

    rows.append({
        "model": model,
        "tensor_category": tensor_type,

        "tensors": len(group),

        "chunks": total_chunks,
        "used_chunks": used_chunks,
        "skipped_chunks": skipped_chunks,

        "pred_none": none,
        "pred_mean": mean,
        "pred_neighbor": neighbor,

        "pred_none_pct": (
            none / used_chunks
            if used_chunks else 0
        ),

        "pred_mean_pct": (
            mean / used_chunks
            if used_chunks else 0
        ),

        "pred_neighbor_pct": (
            neighbor / used_chunks
            if used_chunks else 0
        ),

        "skip_pct": (
            skipped_chunks / total_chunks
            if total_chunks else 0
        ),
    })


model_df = pd.DataFrame(rows)

model_df = model_df.sort_values(
    ["model", "tensor_category"]
)


# ============================================================
# Save tensor-level classification
# ============================================================

data.to_csv(
    OUTPUT_DETAIL,
    index=False
)

global_df.to_csv(
    OUTPUT_GLOBAL,
    index=False
)

model_df.to_csv(
    OUTPUT_MODEL,
    index=False
)


# ============================================================
# Print results
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    220
)

pd.set_option(
    "display.float_format",
    lambda x: f"{x:.4f}"
)


print("\n")
print("=" * 110)
print("GLOBAL PREDICTOR SELECTION BY TENSOR TYPE")
print("=" * 110)

print(
    global_df[
        [
            "tensor_category",
            "tensors",
            "used_chunks",
            "used_chunk_share_pct",
            "pred_none_pct",
            "pred_mean_pct",
            "pred_neighbor_pct",
            "skip_pct",
        ]
    ].to_string(index=False)
)


print("\n")
print("=" * 110)
print("PER-MODEL PREDICTOR SELECTION BY TENSOR TYPE")
print("=" * 110)

print(
    model_df[
        [
            "model",
            "tensor_category",
            "used_chunks",
            "pred_none_pct",
            "pred_mean_pct",
            "pred_neighbor_pct",
            "skip_pct",
        ]
    ].to_string(index=False)
)


print("\n")
print("=" * 110)
print("TENSOR CLASSIFICATION COUNTS")
print("=" * 110)

classification_counts = (
    data
    .groupby("classified_type")
    .agg(
        tensors=("tensor_name", "count"),
        used_chunks=("used_chunks", "sum"),
        chunks=("chunks", "sum"),
    )
    .sort_values(
        "used_chunks",
        ascending=False
    )
)

print(
    classification_counts.to_string()
)


print("\n")
print("=" * 110)
print("OTHER WEIGHTS — RAW TENSOR NAMES")
print("=" * 110)

other = data[data["classified_type"] == "other_weight"]

for model, group in other.groupby("model"):
    print(f"\n[{model}]")
    print(
        group[
            ["tensor_name", "tensor_type", "chunks", "used_chunks",
             "pred_none_pct", "pred_mean_pct", "pred_neighbor_pct"]
        ].to_string(index=False)
    )


print("\n")
print("Saved:")
print(f"  {OUTPUT_GLOBAL}")
print(f"  {OUTPUT_MODEL}")
print(f"  {OUTPUT_DETAIL}")