#include <iostream>
#include <vector>
#include <random>
#include <cstdint>
#include <bitset>
#include "../Lib/EncLib/BinEncoderOB.h"
#include "../Lib/DecLib/BinDecoder.h"
#include "../Lib/CommonLib/ContextModel.h"

// --- assume these classes are already defined ---
// BinEnc: startBinEncoder(), setByteStreamBuf(), encodeBin(), encodeBinTrm(), finish()
// BinDec: setByteStreamBuf(), decodeBin(), decodeBinTrm(), finish()
// StaticCtx: getRLPS(), getMPS()
// TensorType enum: Weight, Bias (or others)

int main() {
    // --- Setup encoder ---
    BinEnc encoder;
    StaticCtx ctx;
    std::vector<uint8_t> bitstream;
    encoder.setByteStreamBuf(&bitstream);
    encoder.startBinEncoder();

    std::mt19937 rng(42); // deterministic seed
    std::uniform_int_distribution<int> ctxDist(0, 12); // 13 contexts
    std::uniform_int_distribution<int> typeDist(0, 1); // two param types

    const int numBins = 30; // number of bins to generate
    std::vector<uint8_t> origBins;
    std::vector<uint8_t> ctxIds;
    std::vector<TensorType> paramTypes;

    // --- Encode bins ---
    for (int i = 0; i < numBins; ++i) {
        uint8_t ctxId = static_cast<uint8_t>(ctxDist(rng));
        TensorType paramType = typeDist(rng) == 0 ? TensorType::Weight : TensorType::Bias;

        // choose bin: 80% chance MPS, 20% LPS
        uint8_t mps = ctx.getMPS(ctxId, paramType);
        std::bernoulli_distribution binDist(0.8);
        uint8_t bin = binDist(rng) ? mps : (1 - mps);

        // save for verification
        origBins.push_back(bin);
        ctxIds.push_back(ctxId);
        paramTypes.push_back(paramType);

        // encode
        encoder.encodeBin(bin, ctx, ctxId, paramType);
    }

    // --- Termination bin for decoder compatibility ---
    encoder.encodeBinTrm(1);

    // finish stream
    encoder.finish();

    // --- Print encoded bytes ---
    std::cout << "Encoded bitstream bytes:\n";
    for (auto b : bitstream)
        std::cout << std::bitset<8>(b) << " ";
    std::cout << "\nOriginal bins:\n";
    for (auto b : origBins)
        std::cout << (int)b << " ";
    std::cout << "\n";

    // --- Setup decoder ---
    BinDec decoder;
    decoder.setByteStreamBuf(bitstream.data());
    decoder.startBinDecoder(); // if needed

    std::vector<uint8_t> decodedBins;

    // --- Decode using the same ctx/paramType sequences ---
    for (int i = 0; i < numBins; ++i) {
        uint8_t ctxId = ctxIds[i];
        TensorType paramType = paramTypes[i];
        uint8_t bin = decoder.decodeBin(ctx, ctxId, paramType);
        decodedBins.push_back(bin);
    }

    // --- Decode termination bin ---
    decoder.decodeBinTrm();

    decoder.finish(); // flush/align decoder if needed

    // --- Verify ---
    std::cout << "Decoded bins:\n";
    for (auto b : decodedBins)
        std::cout << (int)b << " ";
    std::cout << "\n";

    bool match = decodedBins == origBins;
    std::cout << (match ? "SUCCESS: decoded matches original" : "FAIL: mismatch") << std::endl;

    return 0;
}