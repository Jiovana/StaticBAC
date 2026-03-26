import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import csv


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

# ------------------------------------------------------------
# reconstruct tensors into model
# ------------------------------------------------------------
def load_parameters_by_order(model, decoded_meta, folder, qstep_file, bitwidth=8):
    """
    Reconstruct model parameters from decoded tensors,
    filtering only the ones matching the desired bitwidth.
    """
    tensor_qsteps = load_qsteps(qstep_file)

    # Filter decoded_meta to only include the requested bitwidth
    filtered_meta = [t for t in decoded_meta if t["bitwidth"] == bitwidth]

    params = list(model.named_parameters())

    if len(params) != len(filtered_meta):
        print("WARNING: number of params differs:",
              len(params), "vs", len(filtered_meta))

    with torch.no_grad():
        for i, t in enumerate(filtered_meta):
            param_name, param_tensor = params[i]

            bin_path = os.path.join(folder, t["filename"])
            tensor = load_tensor(bin_path, t["shape"])

            if param_name not in tensor_qsteps:
                raise ValueError(f"Missing qstep for {param_name}")
            qstep = tensor_qsteps[param_name]
           # qstep = tensor_qsteps.get(param_name, 1.0)
            tensor = tensor.to(torch.float32) * qstep
            tensor = tensor.to(param_tensor.dtype)

            if tensor.shape != param_tensor.shape:
                print("Shape mismatch:", param_name,
                      tensor.shape, param_tensor.shape)
                continue

            param_tensor.copy_(tensor)

    print("Parameters reconstructed.")
    # Optional: quick sanity check
    print("Comparing reconstruct to original...")
    for num, (name, param) in enumerate(model.named_parameters()):
        orig = param.detach().cpu().numpy()
        recon_path = os.path.join("resnet_tensors_decoded", f"tensor_{num}.bin")  # adjust filename
        recon = np.fromfile(recon_path, dtype=np.int32).reshape(orig.shape)
        qstep = tensor_qsteps.get(name, 1.0)  # <--- safe lookup by name
        recon = recon * qstep
        print(name, "max diff:", np.max(np.abs(orig - recon)))

def load_buffers_by_order(model, decoded_meta, folder, qstep_file):

    tensor_qsteps = load_qsteps(qstep_file, target_bits=8)
    buffers = list(model.named_buffers())

    if len(buffers) != len(decoded_meta):
        print("WARNING: number of buffers differs:",
              len(buffers), "vs", len(decoded_meta))

    with torch.no_grad():
        for i, t in enumerate(decoded_meta):
            buffer_name, buffer_tensor = buffers[i]
            bin_path = os.path.join(folder, t["filename"])
            tensor = load_tensor(bin_path, t["shape"])

            # get qstep safely by name
           # qstep = tensor_qsteps.get(buffer_name, 1.0)

            if t["bitwidth"] == 32:
                tensor = tensor.to(torch.float32)
            else:
                if (buffer_name not in tensor_qsteps ):
                    raise ValueError(f"Missing qstep for {buffer_name}")
                qstep = tensor_qsteps[buffer_name]
                tensor = tensor.to(torch.float32) * qstep

            tensor = tensor.to(buffer_tensor.dtype)

            if buffer_tensor.shape == torch.Size([]) and tensor.numel() == 1:
                buffer_tensor.copy_(tensor.view(()))
                continue

            if tensor.shape != buffer_tensor.shape:
                print("Shape mismatch:", buffer_name,
                    tensor.shape, buffer_tensor.shape)
                continue

            buffer_tensor.copy_(tensor)

    print("Buffers reconstructed.")

"""     # Optional: quick sanity check
    print("Comparing reconstruct to original...")
    for num, (name, buf) in enumerate(model.named_buffers()):
        orig = buf.detach().cpu().numpy()
        recon_path = os.path.join("resnet_buffers_decoded", f"tensor_{num}.bin")  # adjust filename
        recon = np.fromfile(recon_path, dtype=np.int32).reshape(orig.shape)
        qstep = tensor_qsteps.get(name, 1.0)  # <--- safe lookup by name
        recon = recon * qstep
        print(name, "max diff:", np.max(np.abs(orig - recon)))
 """

# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():

    torch.set_num_threads(8)

    device = "cpu"
    print("Using device:", device)

    imagenet_val_dir = r"C:\Users\gomes\OneDrive\Documentos\imagenet\ILSVRC\Data\CLS-LOC\val"

    model_name = "resnet50"

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
        raise ValueError("Unknown model")

    model.eval()

    transform = weights.transforms()

    dataset = datasets.ImageFolder(
        root=imagenet_val_dir,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=recommend_batch_size(model_name),
        shuffle=False,
        num_workers=8
    )


    # ------------------------------------------------------------
    # reconstruction
    # ------------------------------------------------------------
    DECODED_FOLDER_PARAM = "resnet_tensors_decoded"
    DECODED_FOLDER_BUFFERS = "resnet_buffers_decoded"
    META_PATH_buffer = os.path.join(DECODED_FOLDER_BUFFERS, "decoded_tensors.meta")
    META_PATH_param = os.path.join(DECODED_FOLDER_PARAM, "decoded_tensors.meta")


    QSTEP_FILE_PARAM = r"C:\Users\gomes\OneDrive\Documentos\GitHub\nncodec2_work\example\compression scripts\multi_model_quant_eval_run5\resnet50_compression.csv"
    QSTEP_FILE_BUFFER = r"C:\Users\gomes\OneDrive\Documentos\GitHub\nncodec2_work\example\compression scripts\multi_model_quant_eval_run5\resnet50_buffers_compression.csv"

    print("Reading parameter metadata...")
    decoded_meta_param = read_decoded_meta(META_PATH_param)

    print("Reconstructing parameters...")
    load_parameters_by_order(
        model,
        decoded_meta_param,
        DECODED_FOLDER_PARAM,
        QSTEP_FILE_PARAM
    )


    print("Reading buffer metadata...")
    decoded_meta_buffer = read_decoded_meta(META_PATH_buffer)

    print("Reconstructing buffers...")
    load_buffers_by_order(
        model,
        decoded_meta_buffer,
        DECODED_FOLDER_BUFFERS,
        QSTEP_FILE_BUFFER
    )

    model.eval()


    print("Model successfully reconstructed!")


    # ------------------------------------------------------------
    # inference
    # ------------------------------------------------------------

    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in tqdm(dataloader, desc=f"Inference {model_name}"):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, pred1 = outputs.topk(1, dim=1)
            top1_correct += (pred1.squeeze() == labels).sum().item()

            _, pred5 = outputs.topk(5, dim=1)
            top5_correct += sum([labels[i] in pred5[i] for i in range(len(labels))])

            total += labels.size(0)

    top1_acc = top1_correct / total * 100
    top5_acc = top5_correct / total * 100

    print(f"{model_name} Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"{model_name} Top-5 Accuracy: {top5_acc:.2f}%")


if __name__ == "__main__":
    main()