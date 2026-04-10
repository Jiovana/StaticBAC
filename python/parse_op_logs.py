import os
import re
import pandas as pd

LOG_DIR = "..\\op_logs"
TENSORMETA_PATH = "..\\models\\vit_b16\\tensors.meta"
OUTPUT_CSV = "op_stats_vit_b16.csv"


# -----------------------------
# PARSE TENSOR META
# -----------------------------
def parse_tensormeta(path):
    tensors = {}

    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) < 6:
            continue

        tid = int(parts[0])
        name = parts[1]
        ttype = parts[2]
        bitwidth = int(parts[3])
        num_dims = int(parts[4])

        dims = list(map(int, parts[5:5+num_dims]))

        num_elements = 1
        for d in dims:
            num_elements *= d

        tensors[tid] = {
            "id": tid,
            "name": name,
            "type": ttype,
            "bitwidth": bitwidth,
            "num_elements": num_elements
        }

    return tensors


# -----------------------------
# PARSE LOG FILE
# -----------------------------
def parse_log_file(path):
    data = {
        "Tensor": -1,
        "add": 0,
        "sub": 0,
        "mul": 0,
        "shift": 0,
        "cmp": 0,
        "branch": 0,
        "mem": 0,
        "regularBins": 0,
        "bypassBins": 0,
        "loops": 0,
    }

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if ":" not in line:
                continue

            try:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()

                if key in data:
                    data[key] = int(float(val))
            except:
                print(f"[WARN] Failed parsing line in {path}: {line}")

    return data


# -----------------------------
# LOAD LOGS (FIXED ORDER)
# -----------------------------
def load_logs(log_dir):
    logs = {}

    files = [f for f in os.listdir(log_dir) if f.endswith(".txt")]

    # sort by tensor index
    files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

    for fname in files:
        tid = int(re.findall(r'\d+', fname)[0])
        path = os.path.join(log_dir, fname)

        data = parse_log_file(path)

        total_bins = data["regularBins"] + data["bypassBins"]

        if total_bins == 0:
            print(f"[WARN] Skipping empty log: {fname}")
            continue

        logs[tid] = data

    return logs


# -----------------------------
# MERGE + METRICS
# -----------------------------
def build_dataframe(meta, logs):
    rows = []

    for tid, log in logs.items():

        if tid not in meta:
            print(f"[WARN] Missing meta for tensor {tid}")
            continue

        m = meta[tid]

        total_bins = log["regularBins"] + log["bypassBins"]

        total_ops = (
            log["add"] + log["sub"] + log["mul"] +
            log["shift"] + log["cmp"] + log["branch"]
        )

        num_elements = m["num_elements"]

        # SAFETY
        if num_elements == 0 or total_bins == 0:
            continue

        row = {
            "id": tid,
            "name": m["name"],
            "type": m["type"],
            "bitwidth": m["bitwidth"],
            "num_elements": num_elements,

            "regularBins": log["regularBins"],
            "bypassBins": log["bypassBins"],
            "totalBins": total_bins,

            "ops_total": total_ops,
            "mem": log["mem"],
            "branches": log["branch"],
            "loops": log["loops"],

            # NORMALIZED (now correct)
            "bins_per_weight": total_bins / num_elements,
            "ops_per_bin": total_ops / total_bins,
            "branches_per_bin": log["branch"] / total_bins,
            "mem_per_bin": log["mem"] / total_bins,
            "ops_per_weight": total_ops / num_elements,
        }

        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# SUMMARY (IMPROVED)
# -----------------------------
def print_summary(df):

    print("\n==== GLOBAL AVERAGES ====")
    print(df.mean(numeric_only=True))

    print("\n==== BY BITWIDTH ====")
    print(df.groupby("bitwidth").mean(numeric_only=True))

    print("\n==== BY TYPE ====")
    print(df.groupby("type").mean(numeric_only=True))

    print("\n==== SANITY CHECK ====")
    print("Mean bins/weight:", df["bins_per_weight"].mean())
    print("Mean ops/bin:", df["ops_per_bin"].mean())


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    meta = parse_tensormeta(TENSORMETA_PATH)
    logs = load_logs(LOG_DIR)

    df = build_dataframe(meta, logs)

    df.to_csv(OUTPUT_CSV, index=False)

    print_summary(df)

    print("\nSaved CSV to:", OUTPUT_CSV)