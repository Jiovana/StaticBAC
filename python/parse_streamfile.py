file = "efficientnet_b7_bitstream.bin"

with open(file,"rb") as f:
    data = f.read()
        
with open("stream.hex","w") as f:
    for b in data:
        f.write(f"{b:02x}\n")