# StaticBAC – Neural Network Tensor Compression with Static Binary Arithmetic Coding

## Overview

**StaticBAC** is a research-oriented tool for compressing neural network models using **static binary arithmetic coding (BAC)**.
Its main target is to compress already quantized models in INT8 (preferrably). It supports from INT4, to INT32, in steps of 4. 
Internally, the software uses INT32 for all variables. 

The research pipeline:
1. Extract tensors from a trained model (PyTorch / HuggingFace / Torchvision)
2. Quantize weights and biases to fixed-point representations
3. Encode tensors into binary streams using a static entropy model
4. Optionally decode the bitstream to reconstruct the model

This framework is designed for:
- Studying entropy coding efficiency on NN weights
- Evaluating compression vs. distortion trade-offs
- Hardware-oriented research (e.g., CABAC-like implementations)

---

## Features

- Static **binary arithmetic coding (BAC)** 
- Separate **encode / decode modes**
- Metadata-driven reconstruction
- Tensor-level statistics and performance reporting

---

## Repository Structure


StaticBAC/
│
├── source/ # C++ source code (encoder/decoder)
├── python/ # Python preprocessing (model export & quantization)
├── build/ # Build directory
├── models/ # Input models
└── README.md

Main file is test_model_args.cpp
---

## Requirements

### C++ (Core codec)
- C++17 compatible compiler
- CMake ≥ 3.10

### Python (Model export)
- Python ≥ 3.8
- Required packages:
    -- numpy
    -- torch
    -- torchvision
    -- transformers
    -- tqdm


---

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
python create_meta.py \
    --model resnet50 \
    --source torchvision \
    --weights ResNet50_Weights.DEFAULT \
    --out_dir ./models/resnet50
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

**Encode**
./StaticBAC \
    --encode \
    --binaries ./models/resnet50/binaries \
    --meta ./models/resnet50/tensor.meta \
    --bitstream output.bin
**Decode**
./StaticBAC \
    --decode \
    --bitstream output.bin \
    --out_dir ./decoded_model
**Encode + Decode**
./StaticBAC \
    --encode --decode \
    --binaries ./models/resnet50/binaries \
    --meta ./models/resnet50/tensor.meta \
    --bitstream output.bin \
    --out_dir ./decoded_model

### Metadata Format

The tensor.meta file describes all tensors:

numTensors N

id name type bitwidth dims shape... qstep
Example
0 layer1.weight weight 8 4 64 3 7 7 0.0231

This enables:
- Correct reconstruction of tensor shapes
- Proper dequantization
- Mapping back to model structure

### Quantization Strategy
Weights → 8-bit optimal uniform quantization
Bias / other tensors → 12-bit quantization
Small tensors (<32 elements) → higher precision
Buffers → stored as raw int32 (no quantization)

Quantization step (qstep) is optimized via:

Golden-section search
Mean squared error (MSE) minimization
Performance Metrics

## The tool reports:

Encoding time
Decoding time
Compression ratio
Entropy (bits/symbol)
Throughput (MB/s)


### Notes
Tensor names are preserved to simplify reconstruction
Minimal filename sanitization is recommended (/ and \ replaced) - in create_meta
Decoding reconstructs quantized tensors (not original float values)

### Future Work
* Might update static RLPS 
* Exploit tensor semantics to improve contexts
* Parallel chunk processing to speed up encoder, also important for hardware implementation.
* Change BAC to ANS



### Acknowledgements

This project builds on concepts from:

Arithmetic coding (CABAC)
Neural network compression - NNCodec and DeepCABAC
PyTorch / HuggingFace ecosystems
