#include <iostream>
#include <fstream>
#include <vector>
#include "../EncLib/CABACEncoder.h"  // your header
#include "../EncLib/BinEncoder_simple.h"
#include "../CommonLib/ContextModel.h"
#include "../../StaticCoder.h" 
#include "../CommonLib/TypeDef.h"
#include "../DecLib/CABACDecoder.h"

#include <string>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <map>


#define TENSOR_BIN_DIR "models/bert_tensors_binaries/"
#define META_FILE "models/bert_tensors.meta"



struct CodingStats
{
    uint64_t weights = 0;
    uint64_t rawBits = 0;
};

std::vector<int32_t> read_tensor_bin(const std::string &path)
{
    std::ifstream infile(path, std::ios::binary | std::ios::ate);
    if(!infile.is_open())
    {
        std::cerr << "Failed to open file: " << path << std::endl;
        return {};
    }

    std::streamsize size = infile.tellg();
    infile.seekg(0, std::ios::beg);

    if(size % sizeof(int32_t) != 0)
    {
        std::cerr << "File size is not a multiple of int32_t: " << size << std::endl;
        return {};
    }

    std::vector<int32_t> buffer(size / sizeof(int32_t));
    if(!infile.read(reinterpret_cast<char*>(buffer.data()), size))
    {
        std::cerr << "Error reading file: " << path << std::endl;
        return {};
    }

    return buffer;
}

// ------------------------------------------------------------
// Utility: load metadata + tensors
// ------------------------------------------------------------
bool loadModelTensors(std::vector<TensorMeta>& tensors)
{
    std::ifstream meta(META_FILE);

    if(!meta)
    {
        std::cout << "Failed to open metadata file\n";
        return false;
    }

    std::string tag;
    uint32_t numTensors;

    meta >> tag >> numTensors;

    std::cout << "Loading model tensors: " << numTensors << "\n";

    tensors.resize(numTensors);

    for(uint32_t i = 0; i < numTensors; i++)
    {
        TensorMeta& t = tensors[i];

        meta >> t.tensorId;
        meta >> t.name;

        std::string typeStr;
        meta >> typeStr;

        if(typeStr == "Weight") t.tensorType = TensorType::Weight;
        else if(typeStr == "Bias") t.tensorType = TensorType::Bias;
        else t.tensorType = TensorType::Bias; // in case it is neither, better consider a sensitive tensor 'bias' type

        uint32_t bw;
        meta >> bw;

        t.tensorBitwidth = bitwidthFromLiteral(bw);

        meta >> t.numDims;

        t.shape.resize(t.numDims);

        for(uint32_t d = 0; d < t.numDims; d++)
            meta >> t.shape[d];

        // load bin tensor
        std::string binPath = std::string(TENSOR_BIN_DIR) + t.name + ".bin";

        t.data = read_tensor_bin(binPath);

        if(t.data.empty())
        {
            std::cout << "Failed loading tensor data\n";
            return false;
        }
    }

    return true;
}

// ------------------------------------------------------------
// Utility: compute tensor element count
// ------------------------------------------------------------
uint64_t numel(const TensorMeta& t)
{
    uint64_t n = 1;

    for(auto s : t.shape)
        n *= s;

    return n;
}



// ------------------------------------------------------------
// Utility: tensor type parser
// ------------------------------------------------------------
TensorType parseTensorType(const std::string& name)
{
    if(name.find("bias") != std::string::npos)
        return TensorType::Bias;

    if(name.find("norm") != std::string::npos)
        return TensorType::Bias;

    return TensorType::Weight;
}


// Save decoded TensorMeta in the same format it was encoded (.bin tensors + metadata)

void saveDecodedModel(const std::vector<TensorMeta>& model,
                      const std::string& dir)
{
    std::filesystem::create_directory(dir);

    std::ofstream meta(dir + "/decoded_tensors.meta");

    meta << "numTensors " << model.size() << "\n\n";

    for(const auto& t : model)
    {
        // generate filename from tensorId
        std::string filename = "tensor_" + std::to_string(t.tensorId) + ".bin";
        std::string path = dir + "/" + filename;

        std::ofstream out(path, std::ios::binary);

        if (!out)
        {
            std::cerr << "Failed to write: " << path << "\n";
            continue;
        }

        out.write(reinterpret_cast<const char*>(t.data.data()),
                  t.data.size() * sizeof(int32_t));

        out.close();

        meta << t.tensorId << " "
             << filename << " "
             << static_cast<int>(t.tensorType) << " "
             << static_cast<int>(t.tensorBitwidth) << " "
             << t.numDims << " ";

        for(auto s : t.shape)
            meta << s << " ";

        meta << "\n";
    }
    meta.close();

    std::cout << "Decoded tensors saved to: " << dir << "\n";
}

// ------------------------------------------------------------
// Validation
// ------------------------------------------------------------
void validateModel(
    const std::vector<TensorMeta>& original,
    const std::vector<TensorMeta>& decoded)
{
    uint64_t totalMismatch = 0;
    uint64_t totalWeights  = 0;

    for(size_t t = 0; t < original.size(); t++)
    {
        const auto& A = original[t];
        const auto& B = decoded[t];

        uint64_t mism = 0;

        for(size_t i = 0; i < A.data.size(); i++)
        {
            if(A.data[i] != B.data[i])
                mism++;
        }

        totalMismatch += mism;
        totalWeights  += A.data.size();

        if(mism > 0)
        {
            std::cout << "Tensor mismatch: " << A.name
                      << " mismatches=" << mism << "\n";
        }
    }

    std::cout << "\n===== Validation =====\n";

    std::cout << "Total weights: " << totalWeights << "\n";
    std::cout << "Total mismatches: " << totalMismatch << "\n";

    if(totalMismatch == 0)
        std::cout << "Perfect reconstruction\n";
    else
        std::cout << "Reconstruction errors detected\n";
}





// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
int main()
{

    std::map<TensorBitwidth, CodingStats> stats;

    std::vector<TensorMeta> modelTensors;

    if(!loadModelTensors(modelTensors))
        return -1;

    std::cout << "Loaded tensors successfully\n";

    for(const auto& t : modelTensors)
    {
        uint32_t bw = getBitwidthFromEnum(t.tensorBitwidth);

        CodingStats& s = stats[t.tensorBitwidth];

        s.weights += t.data.size();
        s.rawBits += (uint64_t)t.data.size() * bw;
    }


    // --------------------------------------------------
    // ENCODING
    // --------------------------------------------------

    Encoder encoder;

    uint32_t numGtxFlags = 4;

    encoder.initCtxModels(numGtxFlags);

    std::cout << "\n=== Encoding Model ===\n";

    auto encStart = std::chrono::high_resolution_clock::now();

    const std::vector<uint8_t>& bytestream =
        encoder.encodeModel(modelTensors);

    auto encEnd = std::chrono::high_resolution_clock::now();

    double encTime = std::chrono::duration<double>(encEnd - encStart).count();

    std::cout << "Compressed size: "
              << bytestream.size()
              << " bytes\n";


    uint64_t totalRawBits = 0;

    for(auto& [bw, s] : stats)
        totalRawBits += s.rawBits;

    uint64_t compressedBits = bytestream.size() * 8;

    std::ofstream f("bert_model_bitstream.bin", std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytestream.data()),
        bytestream.size());

    std::cout << "\n===== Bitwidth Statistics =====\n";

    for(auto& [bw, s] : stats)
    {
        uint64_t compBits =
            (double)s.rawBits / totalRawBits * compressedBits;

        double bitsPerWeight =
            (double)compBits / s.weights;

        std::cout
            << "Bitwidth "
            << getBitwidthFromEnum(bw)
            << "-bit\n";

        std::cout
            << "  weights: "
            << s.weights << "\n";

        std::cout
            << "  bits/weight: "
            << bitsPerWeight << "\n";
    }

    // --------------------------------------------------
    // DECODING
    // --------------------------------------------------

    std::cout << "\n=== Decoding Model ===\n";

    Decoder decoder;

    decoder.setStream(const_cast<std::vector<uint8_t>&>(bytestream));

    decoder.initCtxModels(numGtxFlags);

    std::vector<TensorMeta> decodedModel;

    auto decStart = std::chrono::high_resolution_clock::now();

    decoder.decodeModel(decodedModel);
   // decoder.finishDecoding();

    auto decEnd = std::chrono::high_resolution_clock::now();

    double decTime = std::chrono::duration<double>(decEnd-decStart).count();

    std::cout << "Decoded tensors: "
              << decodedModel.size()
              << "\n";

    uint64_t originalBytes = 0;

    for(const auto& t : modelTensors)
    {
        uint32_t bw = getBitwidthFromEnum(t.tensorBitwidth);

        originalBytes +=
            (uint64_t)t.data.size() * bw / 8;
    }

    uint64_t compressedBytes = bytestream.size();

    double ratio =
        (double)originalBytes / compressedBytes;

    double encodeMB =
    (double)originalBytes / (1024.0*1024.0);



    // --------------------------------------------------
    // VALIDATION
    // --------------------------------------------------

    validateModel(modelTensors, decodedModel);



    /// save decoded tensormeta
    saveDecodedModel(decodedModel, "bert_decoded");

    std::cout << "\n========== MODEL CODING SUMMARY ==========\n";

    std::cout << "Tensors processed: " << modelTensors.size() << "\n";

    std::cout << "Original size:   "
            << originalBytes / (1024.0*1024.0)
            << " MB\n";

    std::cout << "Compressed size: "
            << compressedBytes / (1024.0*1024.0)
            << " MB\n";

    std::cout << "Compression ratio: "
            << ratio << "\n";

    std::cout << "\nEncoding time: "
            << encTime << " sec\n";

    std::cout << "Decoding time: "
            << decTime << " sec\n";

        std::cout << "Encode speed: "
            << encodeMB / encTime
            << " MB/s\n";
    std::cout << "Decode speed: "
          << encodeMB / decTime
          << " MB/s\n";

    std::cout << "==========================================\n";


    return 0;
}

// BERT RESULTS
//Finished encodig model. Total encoded bits: 1 392 359 557
//Compressed size: 139 067 863 bytes