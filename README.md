# StaticBAC – Neural Network Tensor Compression with Static Binary Arithmetic Coding
# VERSION 2

## Overview

This version introduces updates to the chunk encoding, binarization, and remainder coding, as well as more precise cost estimation and an updated static probability model.  
The focus is on 8-bit quantized parameters.   

The create_meta.py script now quantizes all parameters (except buffers, which are kept 32-bit) to 8-bit and applies RD quantization instead of MSE-only. 
2 new arguments are required for RD quantization: quantizer and lambda_rd. The lambda defines the weight of distortion. It is indicated to use 0.15 or less to reduce reconstruction loss.


## Build Instructions

```bash  
git clone <repo_url>  
cd StaticBAC  
mkdir build  
cd build  
cmake ..  
make  
```

This generates the executable (e.g., StaticBAC).

### Python: Export and Quantize Model

Use the provided script to extract tensors and generate:  

Binary tensor files (.bin)  
Metadata file (tensor.meta)

Example
```bash
python create_meta.py  
    --model resnet50   
    --source torchvision   
    --weights ResNet50_Weights.DEFAULT  
    --out_dir ./models/resnet50
    --quantizer rd
    --lambda_rd 0.15  
``` 
Output:  
models/resnet50/  
├── binaries/  
│   ├── layer1.weight.bin  
│   ├── layer1.bias.bin  
│   └── ...  
└── tensor.meta  

## Running the Codec

The codec supports:  

- Encoding only  
- Decoding only  
- Full encode → decode pipeline  

Examples:  
**Encode**  
./StaticBAC  --encode  --binaries ./models/resnet50/binaries  --meta ./models/resnet50/tensor.meta  --bitstream output.bin  
**Decode**  
./StaticBAC  --decode  --bitstream output.bin  --out_dir ./decoded_model  
**Encode + Decode**  
./StaticBAC  --encode --decode  --binaries ./models/resnet50/binaries  --meta ./models/resnet50/tensor.meta  --bitstream output.bin  --out_dir ./decoded_model  

### Metadata Format

The tensor.meta file describes all tensors in the following pattern:  

numTensors N  

id name type bitwidth dims shape... qstep  
Example  
0 layer1.weight weight 8 4 64 3 7 7 0.0231  

This enables:  
- Correct reconstruction of tensor shapes  
- Proper dequantization  
- Mapping back to model structure  

### Quantization Strategy
All trainable parameters → 8-bit optimal uniform quantization  
Buffers → stored as raw int32 (no quantization)  

Quantization step (qstep) is optimized via:  
* Golden-section search
* Mean squared error (MSE) and entropy minimization, tuned using lambda_rd
* Performance Metrics

## The tool reports:
Encoding time  
Decoding time  
Compression ratio  
Entropy (bits/symbol)  
Throughput (MB/s)  

### Notes
Tensor names are preserved to simplify reconstruction  
Minimal filename sanitization is recommended (/ and \ replaced) - in create_meta  
StaticBAC decoding reconstructs quantized tensors (not original float values, for that you need to save the qsteps)  

### Future Work
* Exploit tensor semantics to improve contexts  
* Parallel chunk processing to speed up the encoder, also important for hardware implementation.  
* Change BAC to ANS  

### Acknowledgements
This project builds on concepts from:  
* Arithmetic coding (CABAC)  
* Neural network compression - NNCodec and DeepCABAC  
* PyTorch / HuggingFace ecosystems  
