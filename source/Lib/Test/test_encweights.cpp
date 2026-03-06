#include <iostream>
#include <fstream>
#include <vector>
#include "../EncLib/CABACEncoder.h"  // your header
#include "../EncLib/BinEncoder_simple.h"
#include "../CommonLib/ContextModel.h"
#include "../../StaticCoder.h" 
#include "../CommonLib/TypeDef.h"
#include "../DecLib/CABACDecoder.h"

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

#include <string>
#include <algorithm>

 int tensorwidth; 

TensorBitwidth getBitwidthFromName(const std::string &tensorName)
{
    std::string lname = tensorName;
    // Convert to lowercase
    std::transform(lname.begin(), lname.end(), lname.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    // Decide type
    if (lname.find("norm") != std::string::npos || lname.find("layernorm") != std::string::npos){
        tensorwidth = 12;
        return TensorBitwidth::BW_12; // norm layers keep 12-bit
    }
    else if (lname.find("dense") != std::string::npos){
        tensorwidth = 12;
        return TensorBitwidth::BW_12;
    }
    else if (lname.find("weight") != std::string::npos) {
        tensorwidth = 8;
        return TensorBitwidth::BW_8;  // weights -> 8-bit
    }
    else if (lname.find("bias") != std::string::npos){
        tensorwidth = 12;
        return TensorBitwidth::BW_12; // biases keep 12-bit
    }
    else{
        tensorwidth = 8;
        return TensorBitwidth::BW_8; // fallback default
    }
}


int main()
{
    // Path to your tensor binary
    std::string tensor_path = "bert_tensors_for_cpp/classifier_weight.bin";
    std::string tensor_name = "classifier_weight.bin";
//\classifier_weight.bin"
//bert_embeddings_LayerNorm_weight.bin
//bert_embeddings_position_embeddings_weight.bin
//bert_encoder_layer_0_attention_output_dense_bias.bin



    // Read the tensor
    std::vector<int32_t> tensor = read_tensor_bin(tensor_path);
    if(tensor.empty()) return -1;

    // Define tensor shape info     
    uint32_t numWeights  = tensor.size();


    std::vector<uint8_t> bytestream;

    TensorBitwidth bitwidth = getBitwidthFromName(tensor_path);

    TensorType tensorType =  // check if name contains "bias" or "norm" to decide type
        (tensor_name.find("bias") != std::string::npos || tensor_name.find("norm") != std::string::npos) ? TensorType::Bias : TensorType::Weight;

    // Instantiate encoder
    Encoder encoder(tensorType, bitwidth);
    

    // Initialize contexts
    uint32_t numGtxFlags = 4;
    encoder.initCtxModels(numGtxFlags);

    std::cout << "=== Encoding Tensor ===\n";
    std::cout << "Num weights: " << numWeights << "\n";
    std::cout << "Bitwidth: " << (int)bitwidth << "\n";

    // Define shape array (example: 1D tensor with size layerWidth)
    //(512, 768)" (2, 768
    //uint32_t shape[1] = { numWeights }; // for this specific tensor it is one dimension. 
    uint32_t shape[2] = { 2, 768 }; // for this specific tensor it is a 2D tensor (1 x numWeights)
    uint32_t numDims = 2;
    // Encode the full tensor
    uint64_t bitsUsed = encoder.encodeLayer(tensor, shape, numDims, tensor_name);

    std::cout << "Bits used: " << bitsUsed << "\n";

    bytestream = encoder.finish();

    std::cout << "Tensor encoded. \n";

     std::cout << "Raw bytes (ideal packed): " << numWeights * tensorwidth / 8 << "\n";
    std::cout << "Compressed bytes: " << bytestream.size() << "\n";
    std::cout << "Compression ratio: "
          << (numWeights * tensorwidth / 8.0) / bytestream.size()
          << "\n";
 

    // =========================
    // === DECODING SECTION ===
    // =========================

    std::cout << "\n=== Decoding Tensor ===\n";

    std::vector<int32_t> decodedTensor;

    Decoder decoder;
    decoder.setStream(bytestream);
    decoder.initCtxModels(numGtxFlags);


    decoder.decodeLayer(decodedTensor);

    decoder.finish(); 

/*    CABACDecoder dec;

    // Start decoder
    dec.startCabacDecoding(bytestream.data());
    dec.initCtxModels(numGtxFlags);

    // Proper shape buffer
    uint32_t decodedShape[8] = {0};   // support up to 8D tensors
    uint32_t decodedNumDims = 0;

    uint64_t decodedbins = 0;

    // Decode header
    decodedbins += dec.decodeTensorHeader(decodedShape, decodedNumDims);

    std::cout << "Decoded tensor header:\n";
    std::cout << "NumDims = " << decodedNumDims << "\n";

    uint32_t decodedNumWeights = 1;

    for (uint32_t d = 0; d < decodedNumDims; d++)
    {
        std::cout << "Shape[" << d << "] = " << decodedShape[d] << "\n";
        decodedNumWeights *= decodedShape[d];
    }

    std::cout << "Total decoded weights = " << decodedNumWeights << "\n";

    // Allocate output
    decodedTensor.resize(decodedNumWeights);

    // Decode weights
    decodedbins += dec.decodeWeights(decodedTensor.data(), decodedNumWeights);

    std::cout << "Total decoded bins: " << decodedbins << "\n";

    dec.terminateCabacDecoding(); */


   std::cout << "Compressed bytes: " << bytestream.size() << "\n";
   for (int i = 0; i < 16 && i < bytestream.size(); i++)
        printf("%02X ", bytestream[i]);
    printf("\n");

    //std::cout << "Bytes read so far: "  << decoder..getBytesRead() << "\n";

    std::cout << "Decoded tensor size: " << decodedTensor.size() << "\n";

    if (decodedTensor.size() != tensor.size())
    {
        std::cout << "ERROR: Size mismatch after decoding!\n";
        std::cout << "Original size = " << tensor.size() << "\n";
        std::cout << "Decoded size  = " << decodedTensor.size() << "\n";
        return -1;
    }

    std::cout << "First 10 original values:\n";
    for (int i = 0; i < 10; i++)
        std::cout << tensor[i] << " ";
    std::cout << "\n";

    std::cout << "First 10 decoded values:\n";
    for (int i = 0; i < 10; i++)
        std::cout << decodedTensor[i] << " ";
    std::cout << "\n";


    // =========================
    // === VALIDATION SECTION ==
    // =========================

    uint64_t mismatchCount = 0;
    double totalAbsError = 0.0;
    double totalRelError = 0.0;
    uint64_t relCount = 0;

    for (size_t i = 0; i < tensor.size(); i++)
    {
        int32_t orig = tensor[i];
        int32_t dec  = decodedTensor[i];

        if (orig != dec)
            mismatchCount++;

        double absErr = std::abs((double)orig - (double)dec);
        totalAbsError += absErr;

        if (orig != 0)
        {
            totalRelError += absErr / std::abs((double)orig);
            relCount++;
        }
    }

    double mismatchPercent =
        100.0 * mismatchCount / tensor.size();

    double avgAbsError =
        totalAbsError / tensor.size();

    double avgRelError =
        (relCount > 0) ? (100.0 * totalRelError / relCount) : 0.0;

    std::cout << "\n=== Validation Results ===\n";
    std::cout << "Mismatch count: " << mismatchCount << "\n";
    std::cout << "Mismatch percentage: " << mismatchPercent << " %\n";
    std::cout << "Average absolute error: " << avgAbsError << "\n";
    std::cout << "Average relative error: " << avgRelError << " %\n";

    if (mismatchCount == 0)
        std::cout << "Perfect reconstruction\n";
    else
        std::cout << "Reconstruction has losses\n";

    mismatchCount = 0;

    for (size_t i = 0; i < tensor.size(); i++)
    {
        int32_t orig = tensor[i];
        int32_t decv = decodedTensor[i];

        if (orig != decv)
        {
            mismatchCount++;

            // Print first 20 mismatches
            if (mismatchCount <= 20)
            {
                std::cout << "Mismatch at index " << i
                        << " | orig=" << orig
                        << " | dec=" << decv
                        << " | diff=" << (orig - decv)
                        << "\n";
            }
        }
    }

    std::cout << "\nTotal mismatches: " << mismatchCount
            << " / " << tensor.size()
            << " (" << (100.0 * mismatchCount / tensor.size()) << "%)\n";

    if (mismatchCount == 0)
        std::cout << "Perfect reconstruction\n";
    else
        std::cout << "Reconstruction incorrect\n"; 

    return 0;

}
