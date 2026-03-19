import numpy as np

# Path to your reconstructed npz file
npz_path = r"C:\Users\gomes\OneDrive\Documentos\GitHub\nncodec2_work\example\compression scripts\multi_model_quant_eval_run5/resnet50_reconstructed_tensors.npz"

# Load the .npz
npz = np.load(npz_path)

# Lists to separate keys
parameter_keys = []
buffer_keys = []

for key in npz.files:
    # Original model key without bits prefix
    orig_name = "__".join(key.split("__")[1:])  # remove "b8__" or "b16__" prefix
    # Heuristic: buffers often include "num_batches_tracked"
    if "num_batches_tracked" in orig_name or "running_" in orig_name:
        buffer_keys.append(orig_name)
    else:
        parameter_keys.append(orig_name)

print(f"Total keys in .npz: {len(npz.files)}")
print(f"Parameter keys: {len(parameter_keys)}")
print(f"Buffer keys: {len(buffer_keys)}\n")

print("Sample buffer keys:")
for k in buffer_keys[:20]:  # show first 20 buffer keys
    print(" -", k)