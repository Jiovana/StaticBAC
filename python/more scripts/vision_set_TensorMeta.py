# multi_model_quant_eval_uniform_with_buffers.py
import os
from tqdm import tqdm
import numpy as np
import torchvision.models as tvmodels
from nncodec.extensions import deepCABAC


BIN_DIR = "vit_tensors_binaries"
META = "vit_tensors.meta"

os.makedirs(BIN_DIR, exist_ok=True)

# Models to test: keys -> loader lambdas
MODEL_LOADERS = {
    # "resnet50": lambda: tvmodels.resnet50(weights=tvmodels.ResNet50_Weights.IMAGENET1K_V1),
    # "efficientnet_b0": lambda: tvmodels.efficientnet_b0(weights=tvmodels.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "vit_b16": lambda: tvmodels.vit_b_16(weights=tvmodels.ViT_B_16_Weights.IMAGENET1K_V1),
}

# HDSP defaults for encodeLayer
CHAN_SKIP_DEFAULT = lambda num_ch: np.zeros(num_ch if num_ch>0 else 1, dtype=np.int32)
HDSP_MODE = deepCABAC.HdspMode.TensorOn
HDSP_HIST = np.zeros(256, dtype=np.int8)

# Encoder context model defaults (tweak if desired)
CABAC_UNARY_LEN = 4
PARAM_OPT_FLAG = 1

# === QUANT LAYER DEFAULT ARGUMENTS PER LAYER TYPE ===
QUANTLAYER_ARGS = {
    "weight": dict(
        dq_flag=0,          # URQ
        qpDensity=2,
        qp=-32,
        lambdaScale=0.0,
        maxNumNoRem=10,
        scan_order=0,
        general_profile_idc=0
    ),
}


# ---------------- optimized uniform quantizer ----------------
def optimal_uniform_quant(x, bitwidth, search_steps=40):
    """
    Per-tensor symmetric uniform quantization optimized (fast).
    Returns (q_int32, qstep).
    """
    x = x.astype(np.float32)
    Qmax = (1 << (bitwidth - 1)) - 1

    if x.size == 0 or np.all(x == 0):
        return np.zeros_like(x, dtype=np.int32), 1.0

    std = float(np.std(x))
    if std == 0:
        return np.zeros_like(x, dtype=np.int32), 1.0

    qstep_min = max(std / (1 << (bitwidth + 2)), 1e-12)
    qstep_max = max(std * 4.0, qstep_min * 2.0)

    phi = (1 + np.sqrt(5)) / 2.0
    invphi = 1.0 / phi

    a, b = qstep_min, qstep_max
    c = b - (b - a) * invphi
    d = a + (b - a) * invphi

    def mse_for_q(qstep):
        q = np.clip(np.round(x / qstep), -Qmax, Qmax)
        diff = x - q * qstep
        return float(np.mean(diff * diff))

    fc = mse_for_q(c)
    fd = mse_for_q(d)

    for _ in range(search_steps):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - (b - a) * invphi
            fc = mse_for_q(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + (b - a) * invphi
            fd = mse_for_q(d)

    qstep_opt = (a + b) / 2.0
    q = np.clip(np.round(x / qstep_opt), -Qmax, Qmax).astype(np.int32)
    return q, float(qstep_opt)

def convert_bitdepth(q_int32, bitwidth, unsigned=False):
    """Clip quantized integers to target bitwidth (signed/unsigned)."""
    if unsigned:
        qmin = 0
        qmax = (1 << bitwidth) - 1
    else:
        qmin = -(1 << (bitwidth - 1))
        qmax = (1 << (bitwidth - 1)) - 1
    return np.clip(q_int32, qmin, qmax).astype(np.int32)

def compute_qstep(qp, qpDensity):
    """Replicate the QP -> qStep mapping used in quantLayer."""
    k = 1 << qpDensity
    mul = k + (qp & (k - 1))
    shift = int(qp) >> qpDensity
    qstep = float(mul) * (2.0 ** (shift - qpDensity))
    return qstep

def get_tensor_type(name):
    lname = name.lower()
    if "bias" in lname:
        return "Bias"
    elif "weight" in lname:
        return "Weight"
    else: 
        return "Other"
    
def write_metadata(meta_path, tensors):
    with open(meta_path, "w") as f:
        f.write(f"numTensors {len(tensors)}\n\n")

        for t in tensors:
            shape_str = " ".join(map(str, t["shape"]))
            line = ( f'{t["id"]} ' f'{t["name"]} ' f'{t["type"]} ' f'{t["bitwidth"]} ' f'{len(t["shape"])} ' f'{shape_str} ' f'{t["qstep"]}\n' )
            f.write(line)


# ---------------- main loop per model ----------------
for model_key, loader in MODEL_LOADERS.items():
    print(f"\n=== Processing model: {model_key} ===")
    try:
        model = loader()
    except Exception as e:
        print(f"Failed to load model {model_key}: {e}")
        continue

    model.eval()

    meta_list = []
    tensor_id = 0
    

    # 1) Process named_parameters (trainable params)
    for name, param in tqdm(list(model.named_parameters()), desc=f"params-{model_key}"):
        arr = param.detach().cpu().numpy().astype(np.float32)
        shape = arr.shape
        numel = int(arr.size)

        lname = name.lower()
        if "bias" in lname:
            layer_type = "bias"
        elif "weight" in lname:
            layer_type = "weight"
        else:
            layer_type = "other"


        bin_len = 5 
        qIndex = np.zeros_like(arr, dtype=np.int32)
        qstep = 1.0
        clip_bit = 8

        if numel < 32: # do not use optimal search.. too small
            clip_bit = 12
            qstep = np.max(np.abs(arr)) / (2**(clip_bit-1) - 1 + 1e-8)
            qIndex = np.round(arr / qstep)
            
        elif layer_type == "weight" and numel >= 10000:
            # Use DeepCABAC quantLayer for weights
            clip_bit = 8
            qa = QUANTLAYER_ARGS[layer_type]
            enc_q = deepCABAC.Encoder()
            try:
                enc_q.initCtxModels(bin_len, PARAM_OPT_FLAG)
                qp_return = enc_q.quantLayer(
                    arr.astype(np.float32),
                    qIndex,
                    qa["dq_flag"],
                    qa["qpDensity"],
                    qa["qp"],
                    qa["lambdaScale"],
                    bin_len,
                    qa["scan_order"],
                    qa["general_profile_idc"]
                )
                # prefer qp_return if it is meaningful (non-zero), else fallback to qa["qp"]
                qp_used = int(qp_return) if qp_return is not None and qp_return != 0 else int(qa["qp"])
                qstep = compute_qstep(qp_used, qa["qpDensity"])
            except Exception as e:
                print("QuantLayer failed for", name, 8, ":", e)
                continue
        else:
            clip_bit = 12
            # Use fast optimized uniform quant for bias/norm/other
            qIndex, qstep = optimal_uniform_quant(arr, 12, search_steps=50)

        if qstep > 0.1:
            print("WARNING large qstep:", name, qstep)

        #choose unsigned if possible
        unsigned_possible = np.all(qIndex >= 0)
        q_clipped = convert_bitdepth(qIndex, clip_bit, unsigned=unsigned_possible)

        # prepare for encoding (deepCABAC expects int32)
        q_for_enc = np.ascontiguousarray(q_clipped.astype(np.int32))


        recon = qIndex.astype(np.float32) * qstep
        err = np.abs(arr - recon)

        print(name, "max", err.max(), "mean", err.mean())

        # save tensor binary
        safe_name = name.replace(".","_")
        tensor_file = f"{name}.bin"
        tensor_path = os.path.join(BIN_DIR, tensor_file)

        q_for_enc.tofile(tensor_path)

        meta_list.append({
            "id": tensor_id,
            "name": name,
            "type": get_tensor_type(name),
            "bitwidth": clip_bit,
            "shape": list(shape),
            "qstep": qstep
        })
        tensor_id += 1

            
    """ # ------------------- 2) Process named_buffers ------------------- NO QUANTIZATION
    for name, buf in tqdm(list(model.named_buffers()), desc=f"buffers-{model_key}"):
        arr = buf.detach().cpu().numpy()

        if arr.size == 0:
            continue

        shape = list(arr.shape) if len(arr.shape) > 0 else [1]
        if len(arr.shape) == 0:
            arr = arr.reshape(1)

        safe_name = name.replace(".", "_")
        tensor_path = os.path.join(BIN_DIR, f"{safe_name}.bin")

        if np.issubdtype(arr.dtype, np.integer):
            arr_to_save = arr.astype(np.int32)
            buf_type = "BufferInt"
            bitwidth = 32
        else:
            arr_to_save = arr.astype(np.float32)
            buf_type = "BufferFloat"
            bitwidth = 32          # full precision — no quantization

        arr_to_save.tofile(tensor_path)

        meta_list.append({
            "id": tensor_id,
            "name": safe_name,
            "type": buf_type,
            "bitwidth": bitwidth,
            "shape": shape,
            "qstep": 1.0
        })
        tensor_id += 1 """



write_metadata(META, meta_list)


print(f"\nSaved binaries to : {BIN_DIR}")
print(f"Saved metadata to  : {META}")
print(f"Total tensors      : {len(meta_list)}")
