import os
import numpy as np
import pandas as pd

folder = r"C:\Users\Jiovana\Documents\Docs new\Github\StaticBAC\StaticBAC\models\vgg19\binaries"

summary = []

for fn in sorted(os.listdir(folder)):
    if not fn.endswith(".bin"):
        continue

    path = os.path.join(folder, fn)

    x = np.fromfile(path, dtype=np.int32)

    if x.size == 0:
        continue

    values, counts = np.unique(x, return_counts=True)
    p = counts / counts.sum()

    entropy = -(p * np.log2(p)).sum()

    summary.append({
        "tensor": fn,
        "numel": x.size,
        "min": x.min(),
        "max": x.max(),
        "mean": x.mean(),
        "std": x.std(),
        "mean_abs": np.mean(np.abs(x)),
        "zeros_%": 100*np.mean(x==0),
        "plusminus1_%":100*np.mean(np.abs(x)==1),
        "unique": len(values),
        "entropy": entropy
    })

df = pd.DataFrame(summary)

print(df)

print("\n===== SUMMARY =====")
print(df.describe())

print("\nWeighted entropy:",
      np.average(df["entropy"], weights=df["numel"]))