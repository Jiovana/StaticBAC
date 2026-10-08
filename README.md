# StaticBAC V2 — Neural Network Tensor Compression with Static Binary Arithmetic Coding

StaticBAC is a lightweight C++ codec for compressing quantized neural-network tensors using **static binary arithmetic coding (BAC)**.

**Current version: StaticBAC V2**

StaticBAC V2 extends the original StaticBAC V1 implementation with:

* chunk-level selection among `NONE`, `MEAN`, and `NEIGHBOR` predictors;
* probability-based coding-cost estimation for predictor selection;
* an extended magnitude binarization with multiple greater-than decisions;
* Golomb–Rice coding for the remaining magnitude;
* fixed probability models specialized by tensor type and prediction mode;
* optional chunk skipping when entropy coding is estimated to be inefficient;
* support for MSE and rate-distortion (RD) quantization in the accompanying Python tools.

The V2 implementation corresponds to the StaticBAC V2 study described in the associated journal manuscript.

The original V1 implementation is retained in the [`v1`](../../tree/v1) branch for reproducibility of the earlier conference work.

---

## Overview

StaticBAC separates **model preparation and quantization** from **entropy coding**.

The Python tools prepare the neural-network tensors:

```text
Neural network
      │
      ▼
Tensor extraction
      │
      ▼
8-bit quantization
      │
      ├── MSE quantization
      │
      └── RD quantization
      │
      ▼
Binary tensor files + metadata
      │
      ▼
StaticBAC V2
      │
      ▼
Compressed bitstream
```

The C++ codec operates on the quantized tensor representation and can perform:

* encoding;
* decoding;
* encoding followed by decoding for verification.

StaticBAC preserves tensor metadata such as names, shapes, data types, bitwidths, and quantization steps to facilitate reconstruction.

---

## Repository Structure

The repository is organized approximately as follows:

```text
StaticBAC/
├── CMakeLists.txt
├── src/                  # C++ StaticBAC implementation
├── include/              # C++ headers
├── create_meta.py        # Model extraction and quantization
├── ...                   # Additional Python utilities
├── models/               # Generated model data
└── README.md
```

Generated model binaries and compressed bitstreams are not required to be stored in the repository.

---

## Building StaticBAC

### Requirements

The C++ codec requires:

* C++17-compatible compiler
* CMake
* Make or another supported build system

Build with:

```bash
git clone <repo_url>
cd StaticBAC

mkdir build
cd build

cmake ..
make
```

This produces the `StaticBAC` executable in the build directory.

---

## Model Preparation and Quantization

The `create_meta.py` script extracts the model parameters, quantizes trainable parameters, and generates the binary tensor representation and metadata required by StaticBAC.

Trainable parameters are quantized to **8-bit signed integers**.

Non-trainable buffers are not quantized by the uniform quantization procedure. They are exported with a logical bitwidth of 32 and stored as int32 values.

### MSE quantization

MSE quantization selects the quantization step by minimizing reconstruction error.

```bash
python create_meta.py \
    --model resnet50 \
    --source torchvision \
    --weights ResNet50_Weights.DEFAULT \
    --out_dir ./models/resnet50 \
    --quantizer mse
```

### Rate-distortion quantization

RD quantization minimizes

$$
J = D + \lambda H,
$$

where \(D\) is the normalized mean-squared error and \(H\) is the normalized entropy of the quantized representation.

The parameter `lambda_rd` controls the weight assigned to entropy:

```bash
python create_meta.py \
    --model resnet50 \
    --source torchvision \
    --weights ResNet50_Weights.DEFAULT \
    --out_dir ./models/resnet50 \
    --quantizer rd \
    --lambda_rd 0.15
```

Larger values of `lambda_rd` generally favor lower-entropy representations at the cost of greater reconstruction error. The value `0.15` was used as the common RD operating point in the V2 evaluation.

The quantization step is determined using a golden-section search over the candidate step-size interval.

---

## Generated Model Representation

The model preparation step generates a directory such as:

```text
models/resnet50/
├── binaries/
│   ├── layer1.weight.bin
│   ├── layer1.bias.bin
│   ├── ...
└── tensor.meta
```

The binary files contain the quantized tensor values.

The `tensor.meta` file contains the information required to reconstruct the tensor representation, including:

* tensor identifier;
* tensor name;
* tensor type;
* bitwidth;
* number of dimensions;
* tensor dimensions;
* quantization step.

For example:

```text
0 layer1.weight weight 8 4 64 3 7 7 0.0231
```

The quantization step is required to recover the corresponding floating-point representation:

$$
\hat{x} = q \cdot \mathrm{qstep}.
$$

StaticBAC itself operates on the integer tensor representation. Model dequantization/reconstruction is handled by the associated Python tools.

---

## Encoding and Decoding

After building the codec, the executable can be used for encoding, decoding, or both.

### Encode

```bash
./StaticBAC \
    --encode \
    --binaries ./models/resnet50/binaries \
    --meta ./models/resnet50/tensor.meta \
    --bitstream output.bin
```

### Decode

```bash
./StaticBAC \
    --decode \
    --bitstream output.bin \
    --out_dir ./decoded_model
```

### Encode and decode

```bash
./StaticBAC \
    --encode \
    --decode \
    --binaries ./models/resnet50/binaries \
    --meta ./models/resnet50/tensor.meta \
    --bitstream output.bin \
    --out_dir ./decoded_model
```

The decode operation reconstructs the quantized tensor binaries and associated metadata. Floating-point model reconstruction requires applying the stored quantization steps.

---

## StaticBAC V2 Coding

StaticBAC V2 processes tensors independently and divides them into chunks of up to **2048 elements**.

For each chunk, the encoder evaluates three prediction modes:

```text
NONE
MEAN
NEIGHBOR
```

The corresponding residuals are evaluated using a probability-based coding-cost estimator. The predictor with the lowest estimated cost is selected.

If the estimated coding rate is too close to the original fixed-width representation, the chunk can instead be stored directly using the original bitwidth.

For entropy-coded chunks, V2 uses:

1. significance (`SIG`);
2. sign in bypass mode;
3. greater-than magnitude decisions;
4. Golomb–Rice coding of the remaining magnitude.

The probability models are **static** and are not updated during encoding or decoding.

Separate fixed models are used according to tensor type and prediction mode.

This design avoids runtime probability adaptation while retaining probability information specific to the observed tensor statistics.

---

## Supported Quantization

| Quantization mode | Objective                | Purpose                                      |
| ----------------- | ------------------------ | -------------------------------------------- |
| `mse`             | Minimize normalized MSE  | Lower reconstruction error                   |
| `rd`              | Minimize \(D+\lambda H\) | Trade reconstruction error for lower entropy |

StaticBAC V2 supports 8-bit, 12-bit, 16-bit, and 32-bit tensor representations in the codec, although the V2 evaluation primarily targets **8-bit trainable parameters**.

---

## Reported Metrics

The codec reports coding-performance information including:

* encoding time;
* decoding time;
* compressed bitstream size;
* bits per element (BPE);
* compression gain;
* entropy;
* throughput;
* memory consumption.

Compression gain is defined as

$$
G =
\frac{S_{\mathrm{original}}-S_{\mathrm{compressed}}}
{S_{\mathrm{original}}}
\times 100\%.
$$

BPE denotes the compressed bitstream size normalized by the number of encoded elements.

---

## V1 and V2

The original StaticBAC V1 implementation introduced the use of static BAC for quantized neural-network tensors.

StaticBAC V2 substantially revises the coding design by introducing:

* systematic tensor-statistics analysis;
* improved predictor selection;
* probability-based coding-cost estimation;
* revised magnitude binarization;
* extended magnitude thresholds;
* Golomb–Rice remainder coding;
* refined fixed probability models;
* broader evaluation across neural-network architectures.

The V1 implementation remains available in the [`v1`](../../tree/v1) branch.

---

## Citation

If you use StaticBAC V2 in your work, please cite the associated publication:

```text
[Publication information to be added after publication]
```

For the original StaticBAC V1 implementation, please refer to the corresponding [conference publication](https://ieeexplore.ieee.org/document/11706788).

---

## Acknowledgements

StaticBAC builds on concepts and software from the neural-network compression and arithmetic-coding communities, including:

* CABAC and binary arithmetic coding;
* NNCodec and DeepCABAC;
* PyTorch;
* Hugging Face Transformers.

