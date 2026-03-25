#ifndef __BINENC__
#define __BINENC__

#include <cstdint>
#include  "../CommonLib/ContextModel.h"
#include <iostream>
#include "../CommonLib/TypeDef.h"

class BinEnc
{
public:
    BinEnc  () {}
    ~BinEnc () {}

    void      startBinEncoder      ();
    void      setByteStreamBuf     ( std::vector<uint8_t> *byteStreamBuf );

    uint32_t  encodeBinold            ( uint32_t bin,  const StaticCtx &ctxMdl, uint8_t ctxId,  TensorType paramType   );
    uint32_t  encodeBin            ( uint32_t bin,  const StaticCtx &ctxMdl, uint8_t ctxId,  TensorType paramType   );

    void      entryPointStart      () { m_Range = 256; }

    uint32_t  encodeBinEP          ( uint32_t bin                    );
    uint32_t  encodeBinsEP         ( uint32_t bins, uint32_t numBins );

    void      encodeBinTrm         ( unsigned bin );
    void      finish               (              );
    void      terminate_write      (              );

    void      emit_bit              (uint32_t bit);
    void      emit_pending          (uint32_t bit);
protected:
    void      write_out         ();
private:
    std::vector<uint8_t>   *m_ByteBuf;
    uint32_t                m_Low;          // will only need 10 bits transiently during renorm , 9 at steady state
    uint32_t                m_Range;        // unchanged, stays in [256, 511], 9 bits
    uint32_t                m_OB;           // outstanding bits counter - replaces the NumBufferedBytes
    int32_t                 m_PendingBit;   // -1 = none yet, 0 or 1 - replaces m_BufferedByte
    uint32_t                m_BitAccum;     // shift register accumulating bits into a byte
    uint32_t                m_BitAccumCount; // how many bits are currently in m_BitAccum

};

#endif // !__BINENC__
