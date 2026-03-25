
#ifndef __BINDEC__
#define __BINDEC__

#include "CommonLib/ContextModel.h"
#include <iostream>


class BinDec
{
public:
    BinDec() : m_Bytes( nullptr ), m_ByteStreamPtr( nullptr ) {}
    ~BinDec() {}

public:
    void          startBinDecoder      (                                     );
    void          setByteStreamBuf     ( uint8_t* byteStreamBuf              );

    uint32_t      decodeBinold            ( StaticCtx &ctxMdl, uint8_t ctxId, TensorType paramType );  
    uint32_t      decodeBin            ( StaticCtx &ctxMdl, uint8_t ctxId, TensorType paramType );  

    uint32_t      decodeBinEP          (                                     );
    uint32_t      decodeBinsEP         ( uint32_t numBins                    );

    unsigned      decodeBinTrm();
    void          finish();

    uint32_t      getBytesRead() { return m_BytesRead; }
    void          setBytesRead(uint32_t bytesRead) { m_BytesRead=bytesRead; }
    void          setByteStreamPtr(uint8_t* byteStreamPtr ) { m_ByteStreamPtr = byteStreamPtr; }
    uint8_t*      getByteStreamPtr() {return m_ByteStreamPtr;}

    void         resolvePending(uint32_t bit);

private:
    uint32_t m_Range;
    int32_t  m_BitsNeeded;
    uint32_t m_Value;
    uint32_t m_BytesRead;
    uint8_t *m_Bytes;
    uint8_t *m_ByteStreamPtr;
    int32_t m_PendingBit;   // tracks the last unresolved bit (-1 = none)
    uint32_t m_OB;          // counts outstanding bits like in the encoder
};

#endif // __BINDEC__
