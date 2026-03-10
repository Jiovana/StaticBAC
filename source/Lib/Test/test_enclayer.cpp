#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include "../EncLib/CABACEncoder.h"
#include "../EncLib/BinEncoder_simple.h"
#include "../CommonLib/ContextModel.h"
#include "../../StaticCoder.h" 
#include "../CommonLib/TypeDef.h"
#include "../DecLib/CABACDecoder.h"


#define TENSOR_BIN_DIR "models/bert_tensors_binaries/"
#define META_FILE "models/bert_tensors.meta"
#define CSV_FILE "compression_results2.csv"
namespace fs = std::filesystem;

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


// ---------------------------------------------------
// Main test
// ---------------------------------------------------
int main()
{
    std::vector<TensorMeta> modelTensors;
    if(!loadModelTensors(modelTensors))
    {
        std::cerr << "Failed to load model tensors\n";
        return -1;
    }

    std::cout << "Loaded " << modelTensors.size() << " tensors\n";

   
    Decoder decoder;

    uint32_t numGtxFlags = 4;

    std::ofstream csv(CSV_FILE);
    csv << "param_name,layer_type,shape,numel,bits_quant,raw_bytes,mse_post,nmse_post,header_bits,"
        << "compressed_bytes,compression_ratio,compression_gain_pct,bits_per_element,"
        << "encode_time,decode_time\n";

    // --------------------------------
    // === ENCODING ALL TENSORS ===
    // --------------------------------

    std::vector<std::vector<uint8_t>> encodedStreams(modelTensors.size());
    uint64_t totalRawBytes = 0;
    uint64_t totalCompressedBytes = 0;

    for(size_t i = 0; i < modelTensors.size(); i++)
    {
        TensorMeta& t = modelTensors[i];

        Encoder encoder;

        uint64_t raw_bytes = t.data.size() * getBitwidthFromEnum(t.tensorBitwidth) / 8; // exact bits already in tensorMeta if needed

        uint32_t headerBits;

        totalRawBytes += t.data.size() * getBitwidthFromEnum(t.tensorBitwidth) / 8; // rough estimate
        // encoding
        encoder.initCtxModels(numGtxFlags);
        auto t_enc_start = std::chrono::high_resolution_clock::now();
        uint64_t bitsUsed = encoder.encodeLayer(t, static_cast<uint16_t>(t.tensorId), headerBits);
        encodedStreams[i] = encoder.finishEncoding();
        auto t_enc_end = std::chrono::high_resolution_clock::now();
        double enc_ms = std::chrono::duration<double, std::milli>(t_enc_end - t_enc_start).count();

        uint64_t compressed_bytes = encodedStreams[i].size();
        totalCompressedBytes += encodedStreams[i].size();

        double totalratio = static_cast<double>(totalRawBytes) / static_cast<double>(totalCompressedBytes);

        double ratio = (compressed_bytes>0)? static_cast<double>(raw_bytes)/compressed_bytes : 0;
        double gain_pct = (ratio>0)? (1 - (1/ratio))*100 : 0;
        double bpe = (compressed_bytes>0)? (compressed_bytes*8.0/t.data.size()) : 0;


        std::cout << "[" << i << "] " << t.name 
                  << " | Type: " << ((t.tensorType==TensorType::Weight)?"Weight":"Bias")
                  << " | Bits used: " << bitsUsed
                  << " | Raw bytes: " << raw_bytes
                  << " | Compressed bytes: " << compressed_bytes
                  << " | Ratio: " << ratio
                  << "\n";

        // decoding
        TensorMeta decodedTensor = t;
        decodedTensor.data.clear();
        decodedTensor.data.resize(t.data.size());

        decoder.setStream(encodedStreams[i]);
        decoder.initCtxModels(numGtxFlags);
        auto t_dec_start = std::chrono::high_resolution_clock::now();
        decoder.decodeLayer(decodedTensor);
        decoder.finishDecoding();
        auto t_dec_end = std::chrono::high_resolution_clock::now();
        double dec_ms = std::chrono::duration<double, std::milli>(t_dec_end - t_dec_start).count();

        // ---------- VALIDATION ----------
        double mse = 0.0;
        double nmse = 0.0;
        for(size_t j = 0; j < t.data.size(); j++)
        {
            double diff = static_cast<double>(t.data[j]) - static_cast<double>(decodedTensor.data[j]);
            mse += diff*diff;
            nmse += (t.data[j]!=0) ? diff*diff/(t.data[j]*t.data[j]) : 0;
        }
        mse /= t.data.size();
        nmse = (t.data.size()>0) ? 100.0*nmse/t.data.size() : 0.0;

        // ---------- CSV OUTPUT ----------
        csv << t.name << ","
            << ((t.tensorType==TensorType::Weight)?"Weight":"Bias") << ",\"";

        // Shape as string
        for(size_t d = 0; d < t.shape.size(); d++)
        {
            csv << t.shape[d];
            if(d+1 < t.shape.size()) csv << "x";
        }
        csv << "\","
            << t.data.size() << ","
            << getBitwidthFromEnum(t.tensorBitwidth) << ","
            << raw_bytes << ","
            << mse << ","
            << nmse << ","
            << headerBits << ","
            << compressed_bytes << ","
            << ratio << ","
            << gain_pct << ","
            << bpe << ","
            << enc_ms << ","
            << dec_ms
            << "\n";
    }

    csv.close();
    std::cout << " Compression results written to " << CSV_FILE << "\n";

    return 0;
}



    