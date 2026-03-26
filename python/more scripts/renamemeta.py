# Files
in_file = "models/vit_tensors.meta"   # replace with your actual file
out_file = "models/vit_tensors_renamed.meta"

# Simple mapping function
def rename_tensor_name(name: str) -> str:
    # direct replacements
    mapping = {
        "class_token": "class_token",
        "conv_proj_weight": "conv_proj.weight",
        "conv_proj_bias": "conv_proj.bias",
        "encoder_pos_embedding": "encoder.pos_embedding",
        "encoder_ln_weight": "encoder.ln.weight",
        "encoder_ln_bias": "encoder.ln.bias",
        "heads_head_weight": "heads.head.weight",
        "heads_head_bias": "heads.head.bias",
    }

    if name in mapping:
        return mapping[name]

    # encoder layer replacements
    import re
    m = re.match(r"encoder_layers_encoder_layer_(\d+)_(.*)", name)
    if m:
        layer = m.group(1)
        rest = m.group(2)
        rest = rest.replace("_ln_1_weight", "ln_1.weight")
        rest = rest.replace("_ln_1_bias", "ln_1.bias")
        rest = rest.replace("_ln_2_weight", "ln_2.weight")
        rest = rest.replace("_ln_2_bias", "ln_2.bias")
        rest = rest.replace("self_attention_in_proj_weight", "self_attention.in_proj_weight")
        rest = rest.replace("self_attention_in_proj_bias", "self_attention.in_proj_bias")
        rest = rest.replace("self_attention_out_proj_weight", "self_attention.out_proj_weight")
        rest = rest.replace("self_attention_out_proj_bias", "self_attention.out_proj_bias")
        rest = re.sub(r"mlp_(\d+)_weight", r"mlp.\1.weight", rest)
        rest = re.sub(r"mlp_(\d+)_bias", r"mlp.\1.bias", rest)
        return f"encoder.layers.encoder_layer_{layer}.{rest}"

    # default: return as-is
    return name

# Process file
with open(in_file, "r") as f:
    lines = f.readlines()

header = lines[0]
tensor_lines = lines[1:]

renamed_lines = []
for line in tensor_lines:
    if not line.strip():
        continue
    parts = line.split(maxsplit=2)  # index, type, name+rest
    if len(parts) < 3:
        print(f"Skipping malformed line: {line}")
        continue
    idx, tensor_type, rest_of_line = parts
    # The first token in rest_of_line is the tensor name
    rest_tokens = rest_of_line.split(maxsplit=1)
    tensor_name = rest_tokens[0]
    remaining = rest_tokens[1] if len(rest_tokens) > 1 else ""
    new_name = rename_tensor_name(tensor_name)
    renamed_lines.append(f"{idx} {tensor_type} {new_name} {remaining}\n")

# Save
with open(out_file, "w") as f:
    f.write(header)
    f.writelines(renamed_lines)

print(f"Tensor names renamed and saved to {out_file}")