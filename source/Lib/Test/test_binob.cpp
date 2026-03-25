#include <vector>
#include <iostream>
#include <bitset>
#include "../Lib/EncLib/BinEncoderOB.h"
#include "../Lib/DecLib/BinDecoder.h"

int main()
{
    // ---------------------------
    // 1️⃣ Encode
    // ---------------------------
    BinEnc enc;
    std::vector<uint8_t> bitstream;
    enc.setByteStreamBuf(&bitstream);
    enc.startBinEncoder();

    // 🔥 Simple deterministic sequence
    uint32_t bins[] = {0,0,0,0,0,1,0,1,1,1,1,0,0};
    const int num_bins = sizeof(bins) / sizeof(bins[0]);

    for (int i = 0; i < num_bins; i++)
        enc.encodeBinEP(bins[i]);

    enc.encodeBinTrm(1);
    enc.finish();

    std::cout << "Encoded bitstream bytes:\n";
    for (auto b : bitstream)
        std::cout << std::bitset<8>(b) << " ";
    std::cout << std::endl << std::endl;

    // ---------------------------
    // 2️⃣ Decode
    // ---------------------------
    BinDec dec;
    dec.setByteStreamBuf(bitstream.data());
    dec.startBinDecoder();

    std::vector<uint32_t> decoded_bins;
    for (int i = 0; i < num_bins; i++)
        decoded_bins.push_back(dec.decodeBinEP());

    dec.decodeBinTrm();
    dec.finish();

    // ---------------------------
    // 3️⃣ Compare
    // ---------------------------
    std::cout << "Decoded bins:\n";
    for (auto b : decoded_bins)
        std::cout << b << " ";
    std::cout << std::endl;

    // Check for correctness
    bool success = true;
    for (int i = 0; i < num_bins; i++)
    {
        if (bins[i] != decoded_bins[i])
        {
            success = false;
            break;
        }
    }

    if (success)
        std::cout << " Decoding successful, matches original sequence." << std::endl;
    else
        std::cout << " Decoding mismatch!" << std::endl;

    return 0;
}