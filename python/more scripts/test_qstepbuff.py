import os
import numpy as np
import csv
import torch

DECODED_FOLDER_BUFFERS = "resnet_buffers_decoded"
QSTEP_FILE_BUFFER = r"C:\Users\gomes\OneDrive\Documentos\GitHub\nncodec2_work\example\compression scripts\multi_model_quant_eval_run5\resnet50_buffers_compression.csv"


# load qsteps
tensor_qsteps = {}
with open(QSTEP_FILE_BUFFER, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        tensor_qsteps[row['param_name']] = float(row['qstep'])

# list all buffer binaries
for fname in os.listdir(DECODED_FOLDER_BUFFERS):
    if not fname.endswith(".bin"):
        continue

    bin_path = os.path.join(DECODED_FOLDER_BUFFERS, fname)
    tensor_name = fname.replace(".bin", "")
    
    # load binary as int32
    arr = np.fromfile(bin_path, dtype=np.int32)
    
    # get qstep
    qstep = tensor_qsteps.get(tensor_name, 1.0)
    
    # reconstruct float
    recon = arr.astype(np.float32) * qstep
    
    print(f"{tensor_name}: shape={recon.shape}, min={recon.min()}, max={recon.max()}, mean={recon.mean()}")