
#include <random>
#include <algorithm>
#include <iostream>
#include "BinDecoder.h"
#include "Utils/global_logger.h"


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


void BinDec::setByteStreamBuf( uint8_t* byteStreamBuf )
{
    m_Bytes       = byteStreamBuf;
}


void BinDec::startBinDecoder()
{
    m_BytesRead   = 0;
    m_BitsNeeded  = -8;

    m_Range = 510;

    CHECK( m_Bytes == nullptr, "Bitstream is not initialized!" );

    // Primes the 15-bit Value window from first 2 bytes
    m_Value = 256 * m_Bytes[ 0 ] + m_Bytes[ 1 ];
    m_ByteStreamPtr   = m_Bytes + 2;
    m_BytesRead      += 2;

     g_logger->log(
        "START," +
        std::to_string(m_Range) + "," +
        std::to_string(m_Value) + "," +
         std::to_string(m_Bytes[0]) + "," +
          std::to_string(m_Bytes[1]) + "," +
        std::to_string(m_BitsNeeded) + "," +
        std::to_string(m_BytesRead) + "," +
        std::to_string(*m_ByteStreamPtr) + "," +
        std::to_string(m_ByteStreamPtr - m_Bytes)
    );
}

uint32_t BinDec::decodeBin(StaticCtx &ctxMdl, uint8_t ctxId, TensorType paramType)
{
    //printf("DecodeBin called: range %d value %d \n", m_Range, m_Value );
    uint32_t rlps = ctxMdl.getRLPS(ctxId, paramType);
    uint32_t mps  = ctxMdl.getMPS(ctxId, paramType);

    uint32_t range_before = m_Range;
    uint32_t value_before = m_Value;

    // determine LPS
    bool isLPS = (m_Value < (rlps << 7)); // uses RLPS directly 
    uint32_t rmps = m_Range - rlps; // this is performed in parallel with LPS determination to save time

    // reconstruct bin
    uint32_t bin = isLPS ? (mps ^ 1) : mps;

    // update range
    m_Range = isLPS ? rlps : rmps;

    // update value if LPS
    if (!isLPS) // MPS case subtracts RLPS - no wait 
        m_Value -= (rlps << 7);

    uint32_t n = clz32(m_Range) - 23;

    m_Range <<= n;
    m_Value <<= n;
    m_BitsNeeded += n;

    int refill = 0;

    int bits_needed_renorm = m_BitsNeeded;

    if (m_BitsNeeded >= 0)
    {
        m_Value += (*m_ByteStreamPtr++) << m_BitsNeeded;
        m_BitsNeeded -= 8;
        m_BytesRead++;
        refill = 1;
    }

     g_logger->log(
        "CTX," + 
        std::to_string(ctxId) + "," +  
        std::to_string(static_cast<int>(paramType)) + "," +
        std::to_string(range_before) + "," +
        std::to_string(value_before) + "," +
        std::to_string(rlps) + "," +
        std::to_string(mps) + "," +
        std::to_string(isLPS) + "," +
        std::to_string(bin) + "," +
        std::to_string(n) + "," +
        std::to_string(m_Range) + "," +
        std::to_string(m_Value) + "," +
        std::to_string(m_BitsNeeded) + "," +
        std::to_string(m_BytesRead) + "," +
        std::to_string(*m_ByteStreamPtr) + "," +
        std::to_string(bits_needed_renorm) + "," +
        std::to_string(refill)
    );


    return bin;
}

uint32_t BinDec::decodeBinEP()
{
    uint32_t value_before = m_Value;
    m_Value            += m_Value;
    int refill = 0;
    int bits_needed_before = m_BitsNeeded + 1;
    if (++m_BitsNeeded >= 0)
    {
        m_Value          += (*m_ByteStreamPtr++);
        m_BitsNeeded      = -8;
        m_BytesRead++;
        refill = 1;
    }
    uint32_t bin = 0;
    uint32_t SR  = m_Range << 7;
    if (m_Value >= SR)
    {
        m_Value   -= SR;
        bin        = 1;
    }

    g_logger->log(
        "EP," + 
        std::to_string(value_before) + "," +
        std::to_string(bits_needed_before) + "," +
        std::to_string(bin) + "," +
        std::to_string(SR) + "," +
        std::to_string(m_Range) + "," +
        std::to_string(m_Value) + "," +
        std::to_string(m_BitsNeeded) + "," +
        std::to_string(m_BytesRead) + "," +
        std::to_string(*m_ByteStreamPtr) + "," +
        std::to_string(refill)
    );

    return bin;
}

uint32_t BinDec::decodeBinsEP( uint32_t numBins )
{
    if (m_Range == 256)
    {
        uint32_t remBins = numBins;
        uint32_t bins    = 0;
        while (remBins > 0)
        {
            uint32_t binsToRead = std::min<uint32_t>(remBins, 8); //read bytes if able to take advantage of the system's byte-read function
            uint32_t binMask    = (1 << binsToRead) - 1;
            uint32_t newBins    = (m_Value >> (15 - binsToRead)) & binMask;
            bins                = (bins << binsToRead) | newBins;
            m_Value             = (m_Value << binsToRead) & 0x7FFF;
            remBins            -= binsToRead;
            m_BitsNeeded       += binsToRead;
            if (m_BitsNeeded >= 0)
            {
                m_Value          |= (*m_ByteStreamPtr++) << m_BitsNeeded;
                m_BitsNeeded     -= 8;
                m_BytesRead++;
            }
        }

        return bins;
    }
    uint32_t remBins = numBins;
    uint32_t bins    = 0;
    while (remBins > 8)
    {
        m_Value     = (m_Value << 8) + ((*m_ByteStreamPtr++) << (8 + m_BitsNeeded));
        uint32_t SR =   m_Range << 15;
        m_BytesRead++;

        for (int i = 0; i < 8; i++)
        {
            bins += bins;
            SR  >>= 1;
            if (m_Value >= SR)
            {
                bins++;
                m_Value -= SR;
            }
        }
        remBins -= 8;
    }
    m_BitsNeeded   += remBins;
    m_Value       <<= remBins;
    if (m_BitsNeeded >= 0)
    {
        m_Value      += (*m_ByteStreamPtr++) << m_BitsNeeded;
        m_BitsNeeded -= 8;
        m_BytesRead++;
    }
    uint32_t SR = m_Range << (remBins + 7);
    for (uint32_t i = 0; i < remBins; i++)
    {
        bins += bins;
        SR  >>= 1;
        if (m_Value >= SR)
        {
            bins++;
            m_Value -= SR;
        }
    }

    return bins;
}


unsigned BinDec::decodeBinTrm()
{
    uint32_t range_before = m_Range;
    uint32_t value_before = m_Value;
    m_Range    -= 2;
    unsigned SR = m_Range << 7;
    if( m_Value >= SR ) {
        g_logger->log(
            "TRM," + 
            std::to_string(range_before) + "," +
            std::to_string(value_before) + "," +
            std::to_string(1) + "," +
            std::to_string(m_Range) + "," +
            std::to_string(m_Value) + "," +
            std::to_string(SR)
        );
        return 1;
    }
    else {
        int refill = 0;
        if( m_Range < 256 ) {
            m_Range += m_Range;
            m_Value += m_Value;
            if( ++m_BitsNeeded == 0 ) {
                m_Value      +=  (*m_ByteStreamPtr++);
                m_BitsNeeded  = -8;
                m_BytesRead++;
                refill = 1;
            }
        }

        g_logger->log(
            "TRM," + 
            std::to_string(range_before) + "," +
            std::to_string(value_before) + "," +
            std::to_string(0) + "," +
            std::to_string(m_Range) + "," +
            std::to_string(m_Value) + "," +
            std::to_string(m_BitsNeeded) + "," +
            std::to_string(m_BytesRead) + "," +
            std::to_string(*m_ByteStreamPtr) + "," +
            std::to_string(refill)
        );
        return 0;
    }
}

void BinDec::finish()
{
   // Step back one byte — the last consumed byte should contain the stop bit 
  unsigned lastByte;
  lastByte = *(--m_ByteStreamPtr);
  // after accounting for bits already consumed (m_bitsneeded),
  // the remaning bits should be exactly 1 followed by zeros = 0x80
  // this matches encoder's emitbit(1) + zero pad
  if( ( ( lastByte << ( 8 + m_BitsNeeded ) ) & 0xff ) != 0x80)
  {
    std::cout << "No proper stop/alignment pattern at end of CABAC stream." << std::endl;
  }


//  CHECK( ( ( lastByte << ( 8 + m_bitsNeeded ) ) & 0xff ) != 0x80,
//        "No proper stop/alignment pattern at end of CABAC stream." );
}

