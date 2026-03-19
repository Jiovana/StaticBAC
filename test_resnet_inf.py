import re

import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.serialization

def recommend_batch_size(model_name, device="cpu"):
    if device == "cpu":
        if "resnet" in model_name:
            return 128      # good balance
        if "efficientnet" in model_name:
            return 256
        if "vit" in model_name:
            return 256
    return 64  # safe fallback


def check_reconstruction_errors(model, loaded_tensors):
    """
    Compare reconstructed tensors (loaded_tensors) against the original model.

    Args:
        model: Original pre-trained PyTorch model (state_dict is used for reference)
        loaded_tensors: dict mapping param names -> numpy arrays (reconstructed)

    Returns:
        errors: dict mapping param name -> dict(max_abs, mean_abs)
    """
    errors = {}
    sd = model.state_dict()
    print(f"Names in sd: {sd.keys()}")
    for name, recon in loaded_tensors.items():
        if name not in sd:
            print(f"WARNING: {name} not found in model state_dict, skipping.")
            continue
        orig = sd[name].cpu().numpy().astype(np.float32)
        if orig.shape != recon.shape:
            print(f"WARNING: shape mismatch for {name}: model {orig.shape}, reconstructed {recon.shape}")
            continue
        abs_diff = np.abs(orig - recon)
        max_err = np.max(abs_diff)
        mean_err = np.mean(abs_diff)
        errors[name] = {"max_abs": max_err, "mean_abs": mean_err}

       # print(f"original: {orig} | recon: {recon} \n")

    return errors

def meta_to_torch_name_efficient(name):
    parts = name.split("_")

    # last token is always weight or biass
    param_type = parts[-1]

    # rebuild path
    path = parts[:-1]

    return ".".join(path) + f".{param_type}"

def meta_to_torch_name_vit(meta_name: str) -> str:
    """
    Convert NPZ stored ViT tensor names to PyTorch state_dict keys.
    """

    name = meta_name

    # Top-level replacements
    if name.startswith("class_token"):
        return "class_token"
    if name.startswith("conv_proj_weight"):
        return "conv_proj.weight"
    if name.startswith("conv_proj_bias"):
        return "conv_proj.bias"
    if name.startswith("encoder_pos_embedding"):
        return "encoder.pos_embedding"
    if name.startswith("encoder_ln"):
        return "encoder.ln"
    if name.startswith("heads_head"):
        return "heads.head"

    # Encoder layers
    # Convert encoder_layers_encoder_layer_0_mlp_0_weight -> encoder.layers.encoder_layer_0.mlp.0.weight
    if name.startswith("encoder_layers_encoder_layer_"):
        # Split after the layer number
        rest = name[len("encoder_layers_encoder_layer_"):]
        parts = rest.split("_")

        layer_num = parts[0]
        sub_parts = parts[1:]  # e.g., ln_1_weight, self_attention_in_proj_weight, mlp_0_weight

        # Base prefix
        prefix = f"encoder.layers.encoder_layer_{layer_num}."

        # Map self_attention and mlp
        if sub_parts[0] == "self":
            # self_attention_in_proj_weight -> self_attention.in_proj_weight
            sub_name = "_".join(sub_parts)  # e.g., self_attention_in_proj_weight
            sub_name = sub_name.replace("self_attention_in_proj_weight", "self_attention.in_proj_weight")
            sub_name = sub_name.replace("self_attention_in_proj_bias", "self_attention.in_proj_bias")
            sub_name = sub_name.replace("self_attention_out_proj_weight", "self_attention.out_proj_weight")
            sub_name = sub_name.replace("self_attention_out_proj_bias", "self_attention.out_proj_bias")
            return prefix + sub_name
        elif sub_parts[0] == "ln":
            # ln_1_weight -> ln_1.weight, ln_2_bias -> ln_2.bias
            ln_part = "_".join(sub_parts)  # e.g., ln_1_weight
            ln_part = ln_part.replace("_weight", ".weight").replace("_bias", ".bias")
            return prefix + ln_part
        elif sub_parts[0] == "mlp":
            # mlp_0_weight -> mlp.0.weight, mlp_3_bias -> mlp.3.bias
            mlp_part = "_".join(sub_parts)
            mlp_part = mlp_part.replace("_weight", ".weight").replace("_bias", ".bias")
            mlp_part = mlp_part.replace("mlp_", "mlp.")  # mlp_0 -> mlp.0
            return prefix + mlp_part

    # fallback
    return name

def main():
    torch.set_num_threads(8)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"Using device: {device}")
    imagenet_val_dir = r"C:\Users\gomes\OneDrive\Documentos\imagenet\ILSVRC\Data\CLS-LOC\val"
    
    
   # === Select model ===
    model_name = "vit_b_16"  # or efficientnet_b0, vit_b_16

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)

    elif model_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        model = models.vit_b_16(weights=weights)

    else:
        raise ValueError("Unknown model name")

    model.eval()

    # === Use native transforms ===
    transform = weights.transforms()

    # === Dataset ===
    dataset = datasets.ImageFolder(
        root=imagenet_val_dir,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=recommend_batch_size(model_name, device), shuffle=False, num_workers=8)


    # load reconstructed tensors
    #torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
    #
    npz = np.load("vit_reconstructed.npz")
    #npz_path = r"C:\Users\gomes\OneDrive\Documentos\GitHub\nncodec2_work\example\compression scripts\multi_model_quant_eval_run5/efficientnet_b0_reconstructed_tensors.npz"
    #npz = np.load(npz_path)
    # choose precision: 8 or 16
    bits = 8
    prefix = f"b{bits}__"

    # build dict matching state_dict keys
    loaded = {}

    for k in npz.files:
        # remove "param_000_" or "buffer_000_"
        raw_name = k.split("_", 2)[-1]
        name = meta_to_torch_name_vit(raw_name)

        arr = npz[k].astype(np.float32)
        loaded[name] = arr


    """ for k in npz.files:
        print(k)
        if not k.startswith(prefix):
            continue
        name = k[len(prefix):]            # get original key
        arr = npz[k].astype(np.float32)
        loaded[name] = arr """
    
    print("Loaded tensors:", len(loaded))
    print("Example keys:", list(loaded.keys())[:10])

    sd = model.state_dict()

    # Check reconstruction errors
    errors = check_reconstruction_errors(model, loaded)

    print("=== Reconstruction error summary ===")
    max_errors = []
    mean_errors = []
    for name, e in errors.items():
        print(f"{name}: max={e['max_abs']:.6f}, mean={e['mean_abs']:.6f}")
        max_errors.append(e['max_abs'])
        mean_errors.append(e['mean_abs'])

    print(f"Overall max error: {np.max(max_errors):.6f}")
    print(f"Overall mean error: {np.mean(mean_errors):.6f}")

    # overwrite matching keys
    for name, val in loaded.items():
        if name in sd:
            sd[name] = torch.from_numpy(val).to(dtype=sd[name].dtype)
        else:
            print("WARNING: stored key not in model:", name)

    model.load_state_dict(sd, strict=True)
    model.eval()
   
    print(f"Model {model_name} successfully reconstructed for {bits} bits !")

 

    # === Inference ===
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Inference {model_name}"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # Top-1
            _, pred1 = outputs.topk(1, dim=1)
            top1_correct += (pred1.squeeze() == labels).sum().item()

            # Top-5
            _, pred5 = outputs.topk(5, dim=1)
            top5_correct += sum([labels[i] in pred5[i] for i in range(len(labels))])

            total += labels.size(0)

    top1_acc = top1_correct / total * 100
    top5_acc = top5_correct / total * 100

    print(f"{model_name} Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"{model_name} Top-5 Accuracy: {top5_acc:.2f}%")

if __name__ == "__main__":
    main()
