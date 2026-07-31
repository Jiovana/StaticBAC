import os
import numpy as np
from pathlib import Path

OLD_DIR = r"C:\Users\Jiovana\Documents\Docs new\Github\StaticBAC\StaticBAC\models\resnet50\binaries"
NEW_DIR = r"C:\Users\Jiovana\Documents\Docs new\Github\StaticBAC\StaticBAC\models8\resnet50\binaries"

old_dir = Path(OLD_DIR)
new_dir = Path(NEW_DIR)

old_map = {}

for f in old_dir.glob("*.bin"):
    # convert old underscore format back to lookup key
    key = f.stem.replace("_", ".")
    old_map[key] = f

new_map = {}

for f in new_dir.glob("*.bin"):
    key = f.stem
    new_map[key] = f

common = sorted(set(old_map) & set(new_map))

#common = sorted(set(f.name for f in old_dir.glob("*.bin")) &                set(f.name for f in new_dir.glob("*.bin")))

print(f"Common tensors: {len(common)}")
print()

different = 0

for name in common:

    old = np.fromfile(old_map[name], dtype=np.int32)
    new = np.fromfile(new_map[name], dtype=np.int32)

    if old.size != new.size:
        print(f"{name}")
        print(f"  SIZE DIFFERENT: {old.size} vs {new.size}")
        different += 1
        continue

    equal = np.array_equal(old, new)

    if equal:
        continue

    different += 1

    diff = old.astype(np.int64) - new.astype(np.int64)

    nz = np.count_nonzero(diff)

    print(f"{name}")
    print(f"  Equal: NO")
    print(f"  Different values : {nz}/{old.size} ({100*nz/old.size:.2f}%)")
    print(f"  Mean abs diff    : {np.mean(np.abs(diff)):.4f}")
    print(f"  Max abs diff     : {np.max(np.abs(diff))}")
    print(f"  Old range        : [{old.min()}, {old.max()}]")
    print(f"  New range        : [{new.min()}, {new.max()}]")

    idx = np.flatnonzero(diff)

    print("  First differences:")
    for i in idx[:10]:
        print(f"    {i}: {old[i]} -> {new[i]}")

    print()

print("=" * 60)
print(f"Different tensors: {different}/{len(common)}")