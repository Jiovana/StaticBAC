#include <random>
#include <algorithm>

#include "BinEncoderOB.h"
#include <bitset>
#include <iostream>
//#include "Utils/global_logger.h"
#include <sstream>


#include <cstdint>
#if defined(_MSC_VER)
#include <intrin.h>
#endif


inline uint32_t clz32(uint32_t x)
{
#if defined(_MSC_VER)
    unsigned long pos;
    _BitScanReverse(&pos, x);
    return 31u - pos;
#else
    return __builtin_clz(x);
#endif
}



void BinEnc::startBinEncoder()
{
    printf(" Im inside binencoderOB\n");
    m_Low                = 0;
    m_Range              = 510;
    m_OB                 = 0;
    m_PendingBit         = -1; // no pending bit yet
    m_BitAccum           = 0;
    m_BitAccumCount      = 0;
}


void BinEnc::setByteStreamBuf( std::vector<uint8_t> *byteStreamBuf )
{
    m_ByteBuf = byteStreamBuf;
}


uint32_t BinEnc::encodeBin(uint32_t bin, const StaticCtx &ctxMdl, uint8_t ctxId, TensorType paramType)
{
    printf("EncodeBin called: range %d , low %d , bin:%d\n", m_Range, m_Low, bin);
    uint32_t rlps = ctxMdl.getRLPS(ctxId, paramType);
    uint32_t mps  = ctxMdl.getMPS(ctxId, paramType);
    uint32_t rmps = m_Range - rlps;

    // determine if bin is LPS
    bool isLPS = (bin != mps);
    // update range
    m_Range = isLPS ? rlps : rmps;

    // update low if LPS
    if (isLPS)
        m_Low += rmps;

    // ---- ob renormalization
    // Renorm: shift Low AND Range, then extract carries — all inside the loop
    while (m_Range < 256)
    {
        if (m_Low < 256)
        {
            emit_pending(0); // resolve PB + OB once
        }
        else if (m_Low >= 512)
        {
            emit_pending(1); // resolve PB + OB once
            m_Low -= 512;
        }
        else
        {
            // underflow, just increment outstanding
            m_OB++;
            m_Low -= 256;
        }

        m_Low <<= 1;
        m_Range <<= 1;
    }

   printf("EncodeBin out: range %d , low %d , ob:%d\n", m_Range, m_Low, m_OB);

    return 1;
}

uint32_t BinEnc::encodeBinEP(uint32_t bin)
{
  printf("EncodeBinEP called: range %d , low %d , bin %d\n", m_Range, m_Low, bin);
    m_Low <<= 1;
    if (bin)
        m_Low += 1;   // just add 1, NOT m_Range

    // Emit the LSB immediately via emit_bit
    emit_bit(m_Low & 1u);
    

   printf("EncodeBinEP out: range %d , low %d , ob %d, pb %d\n", m_Range, m_Low, m_OB, m_PendingBit);

    return 0;
}


uint32_t BinEnc::encodeBinsEP( uint32_t bins, uint32_t numBins )
{
    CHECK( bins >= ( 1u << numBins ), printf( "%i can not be coded with %i EP-Bins", bins, numBins ) )

    // Batching by bytes is incompatible with OB carry tracking —
    // each bit must be checked individually for carry ambiguity.
    // Process MSB first to preserve bit order.
    for (int32_t i = static_cast<int32_t>(numBins) - 1; i >= 0; i--)
    {
        encodeBinEP((bins >> i) & 1u);
    }

    return 0;
}


// Pack a single bit into the output byte buffer
void BinEnc::emit_bit(uint32_t bit)
{
    printf("    emit_bit: %d\n", bit);
    // shift in the new bit at the LSB side
    m_BitAccum = (m_BitAccum << 1) | (bit & 1u);
    m_BitAccumCount++;

    if ( m_BitAccumCount == 8){
      m_ByteBuf->push_back(static_cast<uint8_t>(m_BitAccum & 0xFF));
      m_BitAccum = 0;
      m_BitAccumCount = 0;
    }
    
}

// Called when a bit is unambiguously resolved.
// Flushes the pending bit + all outstanding (inverted) bits, then sets new pending.
void BinEnc::emit_pending(uint32_t bit)
{
    printf("  EMIT_PENDING bit=%d OB=%d PB=%d\n",
      bit, m_OB, m_PendingBit);

    if (m_PendingBit == -1)
    {
        m_PendingBit = bit;
        return;
    }

    if (bit == 1)   // carry case: flip everything
    {
        emit_bit(m_PendingBit ^ 1u);               // PB carries → flip it
        for (uint32_t i = 0; i < m_OB; i++)
            emit_bit(m_PendingBit);                 // OB bits (were PB^1) also flip → PB
    }
    else            // no-carry case: emit as confirmed
    {
        emit_bit(m_PendingBit);
        for (uint32_t i = 0; i < m_OB; i++)
            emit_bit(m_PendingBit ^ 1u);
    }

    m_OB         = 0;
    m_PendingBit = bit;
}




void BinEnc::encodeBinTrm(unsigned bin)
{
    m_Range -= 2;

    if (bin)
        m_Low += m_Range;

    while (m_Range < 256)
    {
        if (m_Low >= 512)
        {
            emit_pending(1); // flush pending + OB
            m_Low -= 512;
        }
        else if (m_Low < 256)
        {
            emit_pending(0);
        }
        else
        {
            m_OB++;
            m_Low -= 256;
        }

        m_Low <<= 1;
        m_Range <<= 1;
    }
}

void BinEnc::finish()
{
    // flush pending + outstanding bits
    if (m_PendingBit >= 0)
    {
        emit_bit(m_PendingBit);
        for (uint32_t i = 0; i < m_OB; i++)
            emit_bit(m_PendingBit ^ 1u);
    }

    m_OB = 0;
    m_PendingBit = -1;

    // if m_BitAccum has leftover bits, leave it alone
    if (m_BitAccumCount > 0)
    {
        // pad current byte to 8 bits with zeros and push
        m_BitAccum <<= (8 - m_BitAccumCount);
        m_ByteBuf->push_back(static_cast<uint8_t>(m_BitAccum));
        m_BitAccum = 0;
        m_BitAccumCount = 0;
    }

    // append a **new stop byte** 0x80
    m_ByteBuf->push_back(0x80);
}