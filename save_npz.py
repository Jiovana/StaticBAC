import numpy as np
import os
import torch
import csv
import torchvision.models as models

BITWIDTH_MAP = {
    0:4, 1:8, 2:12, 3:16, 4:20, 5:24, 6:32
}


def read_encoder_meta(path):

    qsteps = {}
    id_to_name = {}

    with open(path) as f:
        lines = f.readlines()

    for line in lines:

        line = line.strip()

        if not line or line.startswith("numTensors"):
            continue

        parts = line.split()

        if len(parts) < 6:
            continue

        tensor_id = int(parts[0])
        name = parts[1]

        dims = int(parts[4])

        # qstep is last field
        qstep = float(parts[5 + dims])

        qsteps[tensor_id] = qstep
        id_to_name[tensor_id] = name

    print(f"Loaded {len(qsteps)} tensors from encoder meta")

    return qsteps, id_to_name

# ------------------------------------------------------------
# read decoded meta
# ------------------------------------------------------------
def read_decoded_meta(path):

    tensors = []

    with open(path) as f:
        lines = f.readlines()

    for line in lines:

        line = line.strip()

        if not line or line.startswith("numTensors"):
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

    return arr.reshape(shape)

# ------------------------------------------------------------
# build npz
# ------------------------------------------------------------
def build_npz(decoded_meta,
              folder,
              qsteps,
              id_to_name,
              prefix,
              apply_qstep=True):

    data = {}

    assert len(qsteps) == len(decoded_meta), f"Mismatch in tensor count! qsteps{len(qsteps)} x meta{len(decoded_meta)} "

    for t in decoded_meta:

        tensor_id = t["idx"]

        if tensor_id not in id_to_name:
            raise ValueError(f"Missing name for tensor ID: {tensor_id}")

        name = id_to_name[tensor_id]

        bin_path = os.path.join(folder, t["filename"])
        tensor = load_tensor(bin_path, t["shape"]).astype(np.float32)

        if apply_qstep and t["bitwidth"] != 32:
            if tensor_id not in qsteps:
                raise ValueError(f"Missing qstep for tensor ID: {tensor_id}")

            tensor *= qsteps[tensor_id]

        key = f"{prefix}_{tensor_id:03d}_{name}"
        data[key] = tensor

        print(f"[ID {tensor_id}] {name} qstep={qsteps[tensor_id]}")
        print(f"Decoded ID {tensor_id} → {name} | shape={tensor.shape}")

    return data


def normalize_a_key(k):
    return k.replace("b8__", "")

def normalize_b_key(k):
    return "_".join(k.split("_")[2:])  # remove param_000_

# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():

    #model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    #model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model = models.vit_b_16(weights= models.ViT_B_16_Weights.IMAGENET1K_V1)


    # paths
    folder_param = "vit_tensors_decoded"
    #folder_param = "efficientnet_tensors_decoded"
    #folder_buffer = "resnet_buffers_decoded"

    encoder_meta_param = "models/vit_tensors.meta"
    #encoder_meta_buffer = "models/resnet_buffers.meta"

    qstep_param, id_to_name = read_encoder_meta(encoder_meta_param)
    #qstep_buffer = read_qsteps_from_meta(encoder_meta_buffer)

    meta_param = read_decoded_meta(os.path.join(folder_param, "decoded_tensors.meta"))
    #meta_buffer = read_decoded_meta(os.path.join(folder_buffer, "decoded_tensors.meta"))


    param_names = [name for name, _ in model.named_parameters()]
    #buffer_names = [name for name, _ in model.named_buffers()]

    print("\n--- BUILDING PARAM NPZ ---")
    param_data = build_npz(
        meta_param,
        folder_param,
        qstep_param,
        id_to_name,
        prefix="param",
        apply_qstep=True
    )

    # ----------------------------------------------------------------
    # Use ORIGINAL buffers directly from the pretrained model
    # This bypasses reconstruction entirely for diagnostic purposes
    # ----------------------------------------------------------------
    print("\n--- USING ORIGINAL (UNCOMPRESSED) BUFFERS ---")
    buffer_data = {}
    for i, (name, buf) in enumerate(model.named_buffers()):
        key = f"buffer_{i:03d}_{name}"
        buffer_data[key] = buf.numpy()
        #print(f"{key} | shape={buf.shape}")

    # merge
    all_data = {}
    all_data.update(param_data)
    all_data.update(buffer_data)

    out_path = "vit_reconstructed.npz"
    np.savez(out_path, **all_data)

    print(f"\nSaved NPZ → {out_path}")

r""" 
    print("\n--- BUILDING BUFFER NPZ ---")
    buffer_data = build_npz(
        meta_buffer,
        folder_buffer,
        qstep_buffer,
        buffer_names,
        prefix="buffer",
        apply_qstep=True   # ← later you can try False
    ) 

    # merge
    all_data = {}
    all_data.update(param_data)
    all_data.update(buffer_data)

    out_path = "reconstructed_model.npz"
    np.savez(out_path, **all_data)

    print(f"\nSaved NPZ → {out_path}") 

    working_npz = r"C:\Users\gomes\OneDrive\Documentos\GitHub\nncodec2_work\example\compression scripts\multi_model_quant_eval_run5\resnet50_reconstructed_tensors.npz"
    a = np.load(working_npz)
    b = np.load("reconstructed_model.npz")


    keys_a = sorted(a.files)
    a_keys_8 = [k for k in a.files if k.startswith("b8__")]
    keys_b = sorted(b.files)

    a_map = {normalize_a_key(k): k for k in a_keys_8}
    b_map = {normalize_b_key(k): k for k in b.files}

    print("A keys:", len(a_map))
    print("B keys:", len(b_map))

    common = set(a_map.keys()) & set(b_map.keys())

    print(f"Matching tensors: {len(common)}")

    for name in sorted(common):
        ka = a_map[name]
        kb = b_map[name]

        if a[ka].shape != b[kb].shape:
            print(f"Shape mismatch: {name} {a[ka].shape} vs {b[kb].shape}")
            continue

        diff = np.max(np.abs(a[ka] - b[kb]))
        print(f"{name} | diff = {diff:.3e}")

    print("\n--- KEYS IN A ---")
    for k in list(a_map)[:20]:
        print(k)

    print("\n--- KEYS IN B ---")
    for k in list(b_map)[:20]:
        print(k)      """           

    


if __name__ == "__main__":
    main()