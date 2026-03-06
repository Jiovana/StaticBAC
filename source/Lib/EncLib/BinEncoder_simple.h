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
protected:
    void      write_out         ();
private:
    std::vector<uint8_t>   *m_ByteBuf;
    uint32_t                m_Low;
    uint32_t                m_Range;
    uint8_t                 m_BufferedByte;
    uint32_t                m_NumBufferedBytes;
    uint32_t                m_BitsLeft;
    static const uint32_t   m_auiGoRiceRange[ 10 ];
};

#endif // !__BINENC__
