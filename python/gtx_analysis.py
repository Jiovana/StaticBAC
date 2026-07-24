import os
import re
import csv
import argparse
from collections import defaultdict

# ----------------------------------------------------------
# Regex
# ----------------------------------------------------------

re_chunk = re.compile(
    r"bitwidth=(\d+).*?"
    r"predictor=(\d+).*?"
    r".*?skip=(\d+)"
)

re_branch = re.compile(
    r"branchflag=([01]),\s*remabslevel=(\d+)"
)

# ----------------------------------------------------------
# Tensor classification
# ----------------------------------------------------------

def classify_tensor(name):

    lname = name.lower()

    if lname.endswith("weight"):
        return "weight"

    if lname.endswith("bias"):
        return "bias"

    return "buffer"

pred_name = {
    0: "NONE",
    1: "MEAN",
    2: "NEIGHBOR"
}

# ----------------------------------------------------------
# Statistics
# ----------------------------------------------------------

# histogram[type][pred][level] = count
histogram = defaultdict(
    lambda: defaultdict(
        lambda: defaultdict(int)
    )
)

# branch statistics
branch_stats = defaultdict(
    lambda: defaultdict(
        lambda: {
            "gtx": 0,
            "rem": 0,
            "levels": []
        }
    )
)

# ----------------------------------------------------------
# Parse file
# ----------------------------------------------------------

def parse_file(path):

    tensor_name = os.path.splitext(
        os.path.basename(path)
    )[0]

    tensor_type = classify_tensor(tensor_name)

    current_predictor = None
    ignore_chunk = False

    with open(path, "r", errors="ignore") as f:

        for line in f:

            m = re_chunk.search(line)

            if m:

                current_predictor = int(m.group(2))

                skip = int(m.group(3))

                ignore_chunk = bool(skip)

                continue

            if ignore_chunk:
                continue

            m = re_branch.search(line)

            if not m:
                continue

            branch = int(m.group(1))
            level = int(m.group(2))

            histogram[tensor_type][current_predictor][level] += 1

            entry = branch_stats[tensor_type][current_predictor]

            entry["levels"].append(level)

            if branch == 0:
                entry["gtx"] += 1
            else:
                entry["rem"] += 1


# ----------------------------------------------------------
# Percentile
# ----------------------------------------------------------

def percentile(sorted_values, p):

    if len(sorted_values) == 0:
        return 0

    idx = int(round((len(sorted_values)-1)*p))

    return sorted_values[idx]

# ----------------------------------------------------------
# Histogram CSV
# ----------------------------------------------------------

def write_histogram(outfile):

    with open(outfile,"w",newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "tensor_type",
            "predictor",
            "residual_level",
            "count",
            "probability",
            "cumulative_probability"
        ])

        for tensor_type in ["weight","bias","buffer"]:

            if tensor_type not in histogram:
                continue

            for pred in [0,1,2]:

                levels = histogram[tensor_type][pred]

                if len(levels)==0:
                    continue

                total = sum(levels.values())

                cumulative = 0

                for level in sorted(levels):

                    count = levels[level]

                    cumulative += count

                    writer.writerow([
                        tensor_type,
                        pred_name[pred],
                        level,
                        count,
                        count/total,
                        cumulative/total
                    ])

# ----------------------------------------------------------
# Branch statistics CSV
# ----------------------------------------------------------

def write_branch_stats(outfile):

    with open(outfile,"w",newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "tensor_type",
            "predictor",

            "gtx_count",
            "rem_count",

            "gtx_probability",
            "rem_probability",

            "mean_level",
            "median_level",

            "p90",
            "p95",
            "p99",

            "maximum"
        ])

        for tensor_type in ["weight","bias","buffer"]:

            if tensor_type not in branch_stats:
                continue

            for pred in [0,1,2]:

                s = branch_stats[tensor_type][pred]

                levels = sorted(s["levels"])

                if len(levels)==0:
                    continue

                total = len(levels)

                mean = sum(levels)/total

                median = percentile(levels,0.50)

                p90 = percentile(levels,0.90)

                p95 = percentile(levels,0.95)

                p99 = percentile(levels,0.99)

                maximum = levels[-1]

                writer.writerow([

                    tensor_type,
                    pred_name[pred],

                    s["gtx"],
                    s["rem"],

                    s["gtx"]/total,
                    s["rem"]/total,

                    mean,
                    median,

                    p90,
                    p95,
                    p99,

                    maximum

                ])

# ----------------------------------------------------------
# GTX Threshold Coverage CSV
# ----------------------------------------------------------

def write_gtx_thresholds(outfile):

    thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16]

    with open(outfile, "w", newline="") as f:

        writer = csv.writer(f)

        header = [
            "tensor_type",
            "predictor",
            "total_symbols"
        ]

        header.extend([f"<= {t}" for t in thresholds])

        writer.writerow(header)

        for tensor_type in ["weight", "bias", "buffer"]:

            if tensor_type not in histogram:
                continue

            for pred in [0, 1, 2]:

                levels = histogram[tensor_type][pred]

                if len(levels) == 0:
                    continue

                total = sum(levels.values())

                row = [
                    tensor_type,
                    pred_name[pred],
                    total
                ]

                for t in thresholds:

                    covered = sum(
                        count
                        for level, count in levels.items()
                        if level <= t
                    )

                    row.append(covered / total)

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

    os.makedirs(args.output,exist_ok=True)

    files = sorted([
        f for f in os.listdir(args.input)
        if f.endswith(".txt")
    ])

    print(f"Found {len(files)} tensor logs.")

    for file in files:

        print(file)

        parse_file(
            os.path.join(args.input,file)
        )

    write_histogram(
        os.path.join(
            args.output,
            "residual_histogram.csv"
        )
    )

    write_branch_stats(
        os.path.join(
            args.output,
            "branch_statistics.csv"
        )
    )

    write_gtx_thresholds(
        os.path.join(
            args.output,
            "gtx_thresholds.csv"
        )
    )

    print("\nDone.")

if __name__ == "__main__":
    main()