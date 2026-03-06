#include <vector>
#include <cstdint>
#include <iostream>

#include "EncLib/BinEncoder_simple.h"
#include "CommonLib/StaticCtx.h"

int main()
{
    BinEnc enc;
    std::vector<uint8_t> bitstream;

    enc.setByteStreamBuf(&bitstream);
    enc.startBinEncoder();

    // Dummy context with fixed probability
    StaticCtx ctx;
    // ctx.initState(0);  // or whatever constructor you have

    // Encode some fake bins
    for (int i = 0; i < 100; i++)
    {
        enc.encodeBin(i & 1, ctx);
    }

    enc.finish();

    std::cout << "Encoded bytes: " << bitstream.size() << "\n";
    for (auto b : bitstream)
        std::cout << std::hex << (int)b << " ";
    std::cout << "\n";

    return 0;
}
