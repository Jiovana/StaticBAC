
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

const uint32_t BinDec::m_auiGoRiceRange[ 10 ] =
{
    6, 5, 6, 3, 3, 3, 3, 3, 3, 3
};


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

    m_Value = 256 * m_Bytes[ 0 ] + m_Bytes[ 1 ];
    m_ByteStreamPtr   = m_Bytes + 2;
    m_BytesRead      += 2;
}


uint32_t BinDec::decodeBinold( StaticCtx &ctxMdl, uint8_t ctxId, TensorType paramType )
{
    uint32_t rlps    = ctxMdl.getRLPS( ctxId, paramType );
    uint32_t mps     = ctxMdl.getMPS( ctxId, paramType );

    uint32_t rmps    = m_Range - rlps;

     // Determine if LPS
    int32_t is_lps = ((int32_t)(rmps + ~(m_Value >> 7))) >> 31;
    // Update range 
    m_Range          = rmps ^ ((rmps ^ rlps) & is_lps);
    // if LPS, update value
    m_Value         -= (rmps << 7) & is_lps;

    // reconstruct bin
    uint32_t bin     = mps ^ (is_lps & 1);

    // renormalize
    uint32_t n = clz32(m_Range) - 23;

    m_Range <<= n;
    m_Value <<= n;
    m_BitsNeeded += n;
    if (m_BitsNeeded >= 0)
    {
        m_Value += (*m_ByteStreamPtr++) << m_BitsNeeded;
        m_BitsNeeded -= 8;
        m_BytesRead++;
    }

    std::ostringstream ss;
    ss << "decodeBin: ctxId=" << (int)ctxId << " paramType=" << (int)paramType << " rlps=" << rlps << " mps=" << mps << " bin=" << bin;
    LOG_LINE(g_logger, ss.str());
    ss .str("");
    ss << "m_Range: " << m_Range << " m_Value: " << m_Value << " m_BitsNeeded: " << m_BitsNeeded;
    LOG_LINE(g_logger, ss.str());


    return bin;
}

uint32_t BinDec::decodeBin(StaticCtx &ctxMdl, uint8_t ctxId, TensorType paramType)
{
    uint32_t rlps = ctxMdl.getRLPS(ctxId, paramType);
    uint32_t mps  = ctxMdl.getMPS(ctxId, paramType);

    std::ostringstream ss;
    ss << "===> Inside decodeBin: start with Range=" << m_Range << " Value=" << m_Value << " Bits needed=" << m_BitsNeeded;
    LOG_LINE(g_logger, ss.str());

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

    // renormalize
    while (m_Range < 256)
    {
        m_Range <<= 1;
        m_Value <<= 1;
        m_BitsNeeded++;

        if (m_BitsNeeded >= 0)
        {
            m_Value += (*m_ByteStreamPtr++) << m_BitsNeeded;
            m_BitsNeeded -= 8;
            m_BytesRead++;
        }
    }

     ss .str("");
    ss << "decodeBin: ctxId=" << (int)ctxId << " rlps=" << rlps << " mps=" << mps
       << " bin=" << bin << " m_Range=" << m_Range << " m_Value=" << m_Value
       << " m_BitsNeeded=" << m_BitsNeeded;
    LOG_LINE(g_logger, ss.str());

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
            std::ostringstream ss;
    ss << "===> Inside decodeBinsEP  Range=" << m_Range << " Value=" << m_Value << " Bits needed=" << m_BitsNeeded << " bins=" << bins;
    LOG_LINE(g_logger, ss.str());

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

    std::ostringstream ss;
    ss << "===> Inside decodeBinsEP  Range=" << m_Range << " Value=" << m_Value << " Bits needed=" << m_BitsNeeded << " bins=" << bins;
    LOG_LINE(g_logger, ss.str());

    return bins;
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
  unsigned lastByte;
  lastByte = *(--m_ByteStreamPtr);
  if( ( ( lastByte << ( 8 + m_BitsNeeded ) ) & 0xff ) != 0x80)
  {
    std::cout << "No proper stop/alignment pattern at end of CABAC stream." << std::endl;
  }


//  CHECK( ( ( lastByte << ( 8 + m_bitsNeeded ) ) & 0xff ) != 0x80,
//        "No proper stop/alignment pattern at end of CABAC stream." );
}

