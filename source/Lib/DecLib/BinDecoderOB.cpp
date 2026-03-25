
#include <random>
#include <algorithm>
#include <iostream>
#include "BinDecoderOB.h"
//#include "Utils/global_logger.h"


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

    // Primes the 15-bit Value window from first 2 bytes — unchanged
    m_Value = 256 * m_Bytes[ 0 ] + m_Bytes[ 1 ];
    m_ByteStreamPtr   = m_Bytes + 2;
    m_BytesRead      += 2;
}



uint32_t BinDec::decodeBin(StaticCtx &ctxMdl, uint8_t ctxId, TensorType paramType)
{
    printf("DecodeBin called: range %d value %d \n", m_Range, m_Value );
    uint32_t rlps = ctxMdl.getRLPS(ctxId, paramType);
    uint32_t mps  = ctxMdl.getMPS(ctxId, paramType);

    uint32_t rmps = m_Range - rlps;

    // determine LPS
    bool isLPS = (m_Value >= (rmps << 7));

    // reconstruct bin
    uint32_t bin = isLPS ? (mps ^ 1) : mps;

    // update range
    m_Range = isLPS ? rlps : rmps;

    // update value if LPS
    if (isLPS)
        m_Value -= (rmps << 7);

    // renormalization with OB tracking
    while (m_Range < 256) {
        m_Range <<= 1;

        // shift in pending bit first
        m_Value <<= 1;
        if (m_PendingBit >= 0) {
            m_Value |= m_PendingBit;
            for (int i = 0; i < m_OB; i++)
                m_Value |= (m_PendingBit ^ 1) << i;
            m_OB = 0;
            m_PendingBit = -1;
        }

        m_BitsNeeded++;
        if (m_BitsNeeded >= 0) {
            m_Value += (*m_ByteStreamPtr++) << m_BitsNeeded;
            m_BitsNeeded -= 8;
            m_BytesRead++;
        }
    }



    return bin;
}

uint32_t BinDec::decodeBinEP()
{
    m_Value            += m_Value;
    if (++m_BitsNeeded >= 0)
    {
        m_Value          += (*m_ByteStreamPtr++);
        m_BitsNeeded      = -8;
        m_BytesRead++;
    }
    uint32_t bin = 0;
    uint32_t SR  = m_Range << 7;
    if (m_Value >= SR)
    {
        m_Value   -= SR;
        bin        = 1;
    }
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

void BinDec::resolvePending(uint32_t bit)
{
    if (m_PendingBit == -1)
    {
        // first pending bit
        m_PendingBit = bit;
        return;
    }

    // consume the previous pending bit
    m_Value = (m_Value << 1) | m_PendingBit;

    // consume outstanding bits (mirror OB)
    for (uint32_t i = 0; i < m_OB; i++)
        m_Value = (m_Value << 1) | (m_PendingBit ^ 1u);

    m_OB = 0;

    // update pending bit
    m_PendingBit = bit;
}


unsigned BinDec::decodeBinTrm()
{
  m_Range    -= 2;
  unsigned SR = m_Range << 7;
  if( m_Value >= SR )
  {
    return 1;
  }
  else
  {
    if( m_Range < 256 )
    {
      m_Range += m_Range;
      m_Value += m_Value;
      if( ++m_BitsNeeded == 0 )
      {
        m_Value      +=  (*m_ByteStreamPtr++);
        m_BitsNeeded  = -8;
        m_BytesRead++;
      }
    }
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

