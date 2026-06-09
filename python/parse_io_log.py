from pathlib import Path

log_file = "io_logs/io_logs.txt"

max_m_value = -1
min_bitsneeded = 999999
max_bitsneeded = -999999

max_ctx_n_shift = -1
max_ep_sr = -1

max_m_value_line = ""
min_bits_line = ""
max_bits_line = ""

max_ctx_n_line = ""
max_ep_sr_line = ""

with open(log_file, "r") as f:
    lines = f.readlines()

# skip first line
for line_num, line in enumerate(lines[1:], start=2):

    line = line.strip()

    if not line:
        continue

    parts = line.split(",")

    rec_type = parts[0]

    try:

        if rec_type == "CTX":

            # CTX fields
            # parts[4]  -> m_value before
            # parts[9]  -> n shift
            # parts[11] -> m_value after
            # parts[12] -> bitsNeeded after

            mval_before = int(parts[4])
            n_shift     = int(parts[9])
            mval_after  = int(parts[11])
            bitsneeded  = int(parts[15])

            # max m_value
            local_max = max(mval_before, mval_after)

            if local_max > max_m_value:
                max_m_value = local_max
                max_m_value_line = line

            # max ctx renorm shift
            if n_shift > max_ctx_n_shift:
                max_ctx_n_shift = n_shift
                max_ctx_n_line = line

            # bitsNeeded stats
            if bitsneeded < min_bitsneeded:
                min_bitsneeded = bitsneeded
                min_bits_line = line

            if bitsneeded > max_bitsneeded:
                max_bitsneeded = bitsneeded
                max_bits_line = line

        elif rec_type == "EP":

            # EP fields
            # parts[1] -> m_value before
            # parts[3] -> SR value
            # parts[5] -> m_value after
            # parts[6] -> bitsNeeded

            mval_before = int(parts[1])
            sr_val      = int(parts[3])
            mval_after  = int(parts[5])
            bitsneeded  = int(parts[2])

            # max m_value
            local_max = max(mval_before, mval_after)

            if local_max > max_m_value:
                max_m_value = local_max
                max_m_value_line = line

            # max EP shift-register compare value
            if sr_val > max_ep_sr:
                max_ep_sr = sr_val
                max_ep_sr_line = line

            # bitsNeeded stats
            if bitsneeded < min_bitsneeded:
                min_bitsneeded = bitsneeded
                min_bits_line = line

            if bitsneeded > max_bitsneeded:
                max_bitsneeded = bitsneeded
                max_bits_line = line

    except (ValueError, IndexError):
        print(f"Skipping malformed line {line_num}")


print("========== RESULTS ==========")

print(f"\nMaximum m_Value : {max_m_value}")
print("Occurred at     :")
print(max_m_value_line)

print(f"\nMaximum CTX renorm shift (N) : {max_ctx_n_shift}")
print("Occurred at                  :")
print(max_ctx_n_line)

print(f"\nMaximum EP SR value : {max_ep_sr}")
print("Occurred at         :")
print(max_ep_sr_line)

print(f"\nMinimum bitsNeeded : {min_bitsneeded}")
print("Occurred at        :")
print(min_bits_line)

print(f"\nMaximum bitsNeeded : {max_bitsneeded}")
print("Occurred at        :")
print(max_bits_line)