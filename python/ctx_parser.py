import os
import re
import csv
import argparse

# ----------------------------------------------------------
# Regex patterns
# ----------------------------------------------------------

re_numweights = re.compile(r"encodeWeightsChunks:\s*numWeights=(\d+)")
re_chunk = re.compile(
    r"bitwidth=(\d+).*?predictor=(\d+).*?bestK=(\d+).*?bitsPerElement=([0-9.]+).*?skip=(\d+)"
)

re_sig = re.compile(r"sigflag=\s*([01]),?\s*sig ctx=(\d+)")
re_sign = re.compile(r"signflag=([01]),?\s*sign ctx=(\d+)")
re_branch = re.compile(r"branchflag=([01])")
re_gtx = re.compile(r"gtx flag=([01]).*?ctx=(\d+)")
re_msb = re.compile(r"msb([1-6])=([01])")


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


# ----------------------------------------------------------
# CSV header
# ----------------------------------------------------------

header = [
    "tensor_name",
    "tensor_type",
    "num_weights",
    "num_chunks",
    "bitwidth",
    "predictor",
    "bestK",
    "bits_per_element",

    "skip_count",
    "skip_total",
    "skip_p1",
]

for ctx in range(13):
    header.extend([
        f"ctx{ctx}_zeros",
        f"ctx{ctx}_ones",
        f"ctx{ctx}_p1"
    ])


# ----------------------------------------------------------
# Parse one tensor log
# ----------------------------------------------------------

def parse_file(path):

    tensor_name = os.path.splitext(os.path.basename(path))[0]

    tensor_type = classify_tensor(tensor_name)

    contexts = {i: [0, 0] for i in range(13)}

    num_weights = 0

    num_chunks = 0

    skip_count = 0

    skip_total = 0

    bitwidth = ""

    predictor = ""

    bestK = ""

    bits_per_element = ""

    with open(path, "r", errors="ignore") as f:

        for line in f:

            m = re_numweights.search(line)
            if m:
                num_weights = int(m.group(1))
                continue

            m = re_chunk.search(line)
            if m:

                bitwidth = int(m.group(1))
                predictor = int(m.group(2))
                bestK = int(m.group(3))
                bits_per_element = float(m.group(4))

                num_chunks += 1
                skip_total += 1

                if int(m.group(5)):
                    skip_count += 1

                continue

            m = re_sig.search(line)
            if m:

                bit = int(m.group(1))
                ctx = int(m.group(2))

                contexts[ctx][bit] += 1

                continue

            m = re_sign.search(line)
            if m:

                bit = int(m.group(1))
                ctx = int(m.group(2))

                contexts[ctx][bit] += 1

                continue

            m = re_branch.search(line)
            if m:

                bit = int(m.group(1))

                contexts[12][bit] += 1

                continue

            m = re_gtx.search(line)
            if m:

                bit = int(m.group(1))
                ctx = int(m.group(2))

                contexts[ctx][bit] += 1

                continue

            for msb, bit in re_msb.findall(line):

                ctx = 5 + int(msb)

                contexts[ctx][int(bit)] += 1

    row = [
        tensor_name,
        tensor_type,
        num_weights,
        num_chunks,
        bitwidth,
        predictor,
        bestK,
        bits_per_element,
        skip_count,
        skip_total,
        skip_count / skip_total if skip_total else 0.0
    ]

    for ctx in range(13):

        zeros = contexts[ctx][0]
        ones = contexts[ctx][1]

        total = zeros + ones

        p1 = ones / total if total else 0.0

        row.extend([zeros, ones, p1])

    return row


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True,
                        help="Folder containing tensor log txt files")

    parser.add_argument("--output", required=True,
                        help="CSV output")

    args = parser.parse_args()

    txt_files = sorted([
        f for f in os.listdir(args.input)
        if f.endswith(".txt")
    ])

    print(f"Found {len(txt_files)} tensor logs")

    with open(args.output, "w", newline="") as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow(header)

        for fname in txt_files:

            print(fname)

            row = parse_file(os.path.join(args.input, fname))

            writer.writerow(row)

    print("\nDone.")


if __name__ == "__main__":
    main()