#include <iostream>
#include <vector>
#include "../EncLib/CABACEncoder.h"  // your header
#include "../EncLib/BinEncoder_simple.h"    // your bin encoder
#include "../CommonLib/ContextModel.h"

int main()
{
    // Prepare a dummy byte stream for the encoder
    std::vector<uint8_t> bytestream;

    // Instantiate the encoder
    CABACEncoder encoder;

    // Start encoding session
    encoder.startCabacEncoding(&bytestream);

    // Initialize contexts: 
    // numGtxFlags = 5 (for example), paramType = 0 (weights), can use 1 for biases later
    uint32_t numGtxFlags = 5;
    
    // Test both weights and biases
    std::vector<int32_t> weightValues  = {0, 1, -1, 5, -7, 15, 250};
    std::vector<int32_t> biasValues    = {0, 1, -1, 3, -2, 6, -100};

    std::cout << "=== Testing Weights ===\n";
    encoder.initCtxMdls(numGtxFlags, 0); // 0 = weight type
    for(auto w : weightValues)
    {
        encoder.encodeWeightDirect(w);
        std::cout << "Encoded weight " << w << " -> bytestream size: " << bytestream.size() << "\n";
        for (auto b : bytestream)
            std::cout << std::hex << (int)b << " ";

    }
     // Finish encoding
    encoder.terminateCabacEncoding();


    std::cout << "\n=== Testing Biases ===\n";
    encoder.initCtxMdls(numGtxFlags, 1); // 1 = bias type
    for(auto b : biasValues)
    {
        encoder.encodeWeightDirect(b);
        std::cout << "Encoded bias " << b << " -> bytestream size: " << bytestream.size() << "\n";
        for (auto b : bytestream)
            std::cout << std::hex << (int)b << " ";

    }

    // Finish encoding
    encoder.terminateCabacEncoding();

    std::cout << "\nTest complete. Final bytestream length: " << bytestream.size() << " bytes.\n";

    return 0;
}
