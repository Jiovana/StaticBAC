import re
import numpy as np

with open("efficientnet_b7_015.txt", "r") as f:
    text = f.read()

entropy = []
zeros = []
mse = []

for line in text.splitlines():
    m = re.search(
        r"Entropy=([0-9.]+).*?Zeros=([0-9.]+)%.*?MSE=([0-9.e+-]+)",
        line
    )
    if m:
        entropy.append(float(m.group(1)))
        zeros.append(float(m.group(2)))
        mse.append(float(m.group(3)))

print("Samples:", len(entropy))
print(f"Entropy avg: {np.mean(entropy):.4f}")
print(f"Zeros avg:   {np.mean(zeros):.4f}%")
print(f"MSE avg:     {np.mean(mse):.6e}")