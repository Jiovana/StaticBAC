# bert_deepcabac_quant_eval_mixed.py
import os
import csv
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification
from nncodec.extensions import deepCABAC

BIN_DIR = "bert_tensor_binaries"
META_FILE = "bert_tensors.meta"

os.makedirs(BIN_DIR, exist_ok = True)

MODEL_NAME = "textattack/bert-base-uncased-SST-2"

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

    # these values are placeholders; uniform-optimizer will override qstep anyway
    "bias": dict(
        dq_flag=1,
        qpDensity=2,
        qp=-50,
        lambdaScale=0.05,
        maxNumNoRem=0,
        scan_order=0,
        general_profile_idc=0
    ),

    "norm": dict(
        dq_flag=1,
        qpDensity=2,
        qp=-50,
        lambdaScale=0.05,
        maxNumNoRem=0,
        scan_order=0,
        general_profile_idc=0
    ),

    "other": dict(
        dq_flag=1,
        qpDensity=2,
        qp=-32,
        lambdaScale=0.0,
        maxNumNoRem=0,
        scan_order=0,
        general_profile_idc=0
    ),
}

# ---------------- utilities ----------------
def compute_qstep(qp, qpDensity):
    """Replicate the QP -> qStep mapping used in quantLayer."""
    k = 1 << qpDensity
    mul = k + (qp & (k - 1))
    shift = int(qp) >> qpDensity
    qstep = float(mul) * (2.0 ** (shift - qpDensity))
    return qstep

def convert_bitdepth(q_int32, bitwidth, unsigned=False):
    """Clip quantized integers to target bitwidth (signed/unsigned)."""
    if unsigned:
        qmin = 0
        qmax = (1 << bitwidth) - 1
    else:
        qmin = -(1 << (bitwidth - 1))
        qmax = (1 << (bitwidth - 1)) - 1
    return np.clip(q_int32, qmin, qmax).astype(np.int32)

# ---------------- optimized uniform quantizer ----------------
def optimal_uniform_quant(x, bitwidth, search_steps=40):
    """
    Fast per-tensor MSE-optimal symmetric uniform quantization.
    Returns (q_int32, qstep).
    """
    x = x.astype(np.float32)
    Qmax = (1 << (bitwidth - 1)) - 1

    # trivial case
    if x.size == 0 or np.all(x == 0):
        return np.zeros_like(x, dtype=np.int32), 1.0

    std = float(np.std(x))
    # if std is tiny, still need small qstep; use dynamic bounds
    if std == 0:
        qstep_opt = 1.0
        q = np.zeros_like(x, dtype=np.int32)
        return q, qstep_opt

    # initial search interval (empirically robust)
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

# ---------------- load model ----------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

tensor_meta_list = []
tensor_id = 0

# ---------------- main loop ----------------
for name, param in tqdm(list(model.named_parameters()), desc="params"):
    arr = param.detach().cpu().numpy().astype(np.float32)
    shape = arr.shape
    numel = int(arr.size)

    lname = name.lower()
    if "norm" in lname or "layernorm" in lname:
        layer_type = "norm"
    elif "dense" in lname:
        layer_type = "dense"
    elif "weight" in lname:
        layer_type = "weight"
    elif "bias" in lname:
        layer_type = "bias"
    else:
        layer_type = "other"

    print(f"Processing layer: {name}, type: {layer_type}, shape: {shape}")

    # variance for NMSE normalization
    var = float(np.var(arr)) if np.var(arr) > 0 else 1.0

   
    bin_len = 5 # number of GTx flags 

    qIndex = np.zeros_like(arr, dtype=np.int32)
    qstep = None
    clip_bit = 8

    if numel < 32: # do not use optimal search.. too small
        clip_bit = 12
        qstep = np.max(np.abs(arr)) / (2**(clip_bit-1) - 1 + 1e-8)
        qIndex = np.round(arr / qstep)
            
    elif  layer_type == "weight":
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

    # Clip to desired bitdepth BEFORE encoding 
    q_clipped = convert_bitdepth(qIndex, clip_bit , unsigned=False)

    # Prepare for encoding (deepCABAC expects int32)
    q_for_enc = np.ascontiguousarray(q_clipped.astype(np.int32))

    # save tensor binary
    safe_name = name.replace(".","_")
    tensor_file = f"{name}.bin"
    tensor_path = os.path.join(BIN_DIR, tensor_file)

    q_for_enc.astype(np.int32).tofile(tensor_path)


    tensor_meta = {
        "id": tensor_id,
        "name": name,
        "type": get_tensor_type(name),
        "bitwidth": clip_bit,
        "shape": list(q_for_enc.shape),
        "qstep": qstep
    }

    tensor_meta_list.append(tensor_meta)
    tensor_id += 1


write_metadata(META_FILE, tensor_meta_list)

print("Saved tensors to:", BIN_DIR)
print("Saved metadata to:", META_FILE)
print("Total tensors:", len(tensor_meta_list))


