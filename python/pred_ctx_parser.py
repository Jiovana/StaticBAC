import os
import re
import csv
import argparse
from collections import defaultdict

# ----------------------------------------------------------
# Regex
# ----------------------------------------------------------

re_chunk_start = re.compile(r"encodeWeightsChunks:\s*numWeights=(\d+)")

re_chunk = re.compile(
    r"bitwidth=(\d+).*?"
    r"predictor=(\d+).*?"
    r"costNone=.*?"
    r"costMean=.*?"
    r"costNeighbor=.*?"
    r"bestK=(\d+).*?"
    r"bitsPerElement=([0-9.]+).*?"
    r"skip=(\d+)"
)

re_sig = re.compile(r"sigflag=\s*([01]),?\s*sig ctx=(\d+)")
re_sign = re.compile(r"signflag=([01]),?\s*sign ctx=(\d+)")
re_branch = re.compile(r"branchflag=([01])")
re_gtx = re.compile(r"gtx flag=([01]).*?ctx=(\d+)")
re_msb = re.compile(r"msb([1-6])=([01])")

# ----------------------------------------------------------
# Tensor type
# ----------------------------------------------------------

def classify_tensor(name):

    lname = name.lower()

    if lname.endswith("weight"):
        return "weight"

    if lname.endswith("bias"):
        return "bias"

    return "buffer"

# ----------------------------------------------------------
# Predictor names
# ----------------------------------------------------------

pred_name = {
    0: "NONE",
    1: "MEAN",
    2: "NEIGHBOR"
}

# ----------------------------------------------------------
# Global statistics
# ----------------------------------------------------------

# context statistics:
#
# stats[type][predictor][ctx] = [zeros, ones]

stats = defaultdict(
    lambda: defaultdict(
        lambda: {i: [0, 0] for i in range(13)}
    )
)

# predictor usage per tensor

usage = {}

# ----------------------------------------------------------
# Parse one file
# ----------------------------------------------------------

def parse_file(path):

    tensor_name = os.path.splitext(os.path.basename(path))[0]
    tensor_type = classify_tensor(tensor_name)

    usage[tensor_name] = {
        "tensor_type": tensor_type,
        "chunks": 0,
        "skip": 0,
        "pred0": 0,
        "pred1": 0,
        "pred2": 0
    }

    current_predictor = None
    ignore_chunk = False

    with open(path, "r", errors="ignore") as f:

        for line in f:

            # -----------------------------
            # new chunk
            # -----------------------------

            m = re_chunk.search(line)

            if m:

                predictor = int(m.group(2))
                skip = int(m.group(5))

                usage[tensor_name]["chunks"] += 1

                if skip:

                    usage[tensor_name]["skip"] += 1
                    ignore_chunk = True

                else:

                    usage[tensor_name][f"pred{predictor}"] += 1
                    current_predictor = predictor
                    ignore_chunk = False

                continue

            if ignore_chunk:
                continue

            # -----------------------------
            # SIG
            # -----------------------------

            m = re_sig.search(line)

            if m:

                bit = int(m.group(1))
                ctx = int(m.group(2))

                stats[tensor_type][current_predictor][ctx][bit] += 1

                continue

            # -----------------------------
            # SIGN
            # -----------------------------

            m = re_sign.search(line)

            if m:

                bit = int(m.group(1))
                ctx = int(m.group(2))

                stats[tensor_type][current_predictor][ctx][bit] += 1

                continue

            # -----------------------------
            # BRANCH
            # -----------------------------

            m = re_branch.search(line)

            if m:

                bit = int(m.group(1))

                stats[tensor_type][current_predictor][12][bit] += 1

                continue

            # -----------------------------
            # GTX
            # -----------------------------

            m = re_gtx.search(line)

            if m:

                bit = int(m.group(1))
                ctx = int(m.group(2))

                stats[tensor_type][current_predictor][ctx][bit] += 1

                continue

            # -----------------------------
            # MSBs
            # -----------------------------

            for msb, bit in re_msb.findall(line):

                ctx = 5 + int(msb)

                stats[tensor_type][current_predictor][ctx][int(bit)] += 1

# ----------------------------------------------------------
# Write predictor usage
# ----------------------------------------------------------

def write_predictor_usage(outfile):

    header = [
        "tensor_name",
        "tensor_type",
        "chunks",
        "used_chunks",
        "skipped_chunks",
        "pred_none",
        "pred_mean",
        "pred_neighbor",
        "pred_none_pct",
        "pred_mean_pct",
        "pred_neighbor_pct",
        "skip_pct"
    ]

    with open(outfile, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(header)

        for tensor in sorted(usage):

            u = usage[tensor]

            used = u["pred0"] + u["pred1"] + u["pred2"]

            writer.writerow([

                tensor,
                u["tensor_type"],

                u["chunks"],
                used,
                u["skip"],

                u["pred0"],
                u["pred1"],
                u["pred2"],

                u["pred0"]/used if used else 0,
                u["pred1"]/used if used else 0,
                u["pred2"]/used if used else 0,

                u["skip"]/u["chunks"] if u["chunks"] else 0

            ])

# ----------------------------------------------------------
# Write context statistics
# ----------------------------------------------------------

def write_context_stats(outfile):

    header = [
        "tensor_type",
        "predictor"
    ]

    for ctx in range(13):

        header.extend([
            f"ctx{ctx}_zeros",
            f"ctx{ctx}_ones",
            f"ctx{ctx}_total",
            f"ctx{ctx}_p1"
        ])

    with open(outfile, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(header)

        for tensor_type in ["weight", "bias", "buffer"]:

            if tensor_type not in stats:
                continue

            for predictor in [0,1,2]:

                row = [

                    tensor_type,
                    pred_name[predictor]

                ]

                for ctx in range(13):

                    z = stats[tensor_type][predictor][ctx][0]
                    o = stats[tensor_type][predictor][ctx][1]

                    total = z + o

                    row.extend([
                        z,
                        o,
                        total,
                        o/total if total else 0
                    ])

                writer.writerow(row)

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing tensor logs"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output folder"
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    files = sorted(
        f for f in os.listdir(args.input)
        if f.endswith(".txt")
    )

    print(f"Found {len(files)} tensor logs.")

    for f in files:

        print(f)

        parse_file(
            os.path.join(args.input, f)
        )

    write_context_stats(
        os.path.join(
            args.output,
            "context_stats_by_predictor.csv"
        )
    )

    write_predictor_usage(
        os.path.join(
            args.output,
            "predictor_usage.csv"
        )
    )

    print("\nDone.")

if __name__ == "__main__":
    main()