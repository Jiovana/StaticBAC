#include <random>
#include <algorithm>

#include "BinEncoder_simple.h"
#include <bitset>
#include <iostream>
#include "Utils/global_logger.h"
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




const uint32_t BinEnc::m_auiGoRiceRange[ 10 ] =
{
    6, 5, 6, 3, 3, 3, 3, 3, 3, 3
};


void BinEnc::startBinEncoder()
{
    m_Low                = 0;
    m_Range              = 510;
    m_BitsLeft           = 23;
    m_NumBufferedBytes   = 0;
}


void BinEnc::setByteStreamBuf( std::vector<uint8_t> *byteStreamBuf )
{
    m_ByteBuf = byteStreamBuf;
}

//read only context
uint32_t BinEnc::encodeBinold( uint32_t bin, const StaticCtx &ctxMdl, uint8_t ctxId, TensorType paramType )
{
  uint8_t rlps = ctxMdl.getRLPS( ctxId, paramType );
  m_Range -= rlps;

  uint32_t mps = ctxMdl.getMPS( ctxId, paramType );

  if (bin == mps)
  {
    if (m_Range < 256)
    {
      m_Range += m_Range;
      m_Low += m_Low;
      m_BitsLeft -= 1;
      if (m_BitsLeft < 12){
        { 
        write_out();
        }
      }
    }
  }
  else
  {
    uint32_t n = clz32(rlps) - 23;
    m_Low += m_Range;
    m_Range = rlps << n;
    m_Low <<= n;
    m_BitsLeft -= n;
    if (m_BitsLeft < 12){
      { 
      write_out();
      }
    }
  }
  
  std::ostringstream ss;
  ss << "m_Range: " << m_Range << " m_Low: " << m_Low << " m_BitsLeft: " << m_BitsLeft;
  LOG_LINE(g_logger, ss.str());
  ss.str("");
  ss << "==> encodeBin called with bin: " << bin << " ctxId: " << (int)ctxId << " paramType: " << (int)paramType << " rlps: " << (uint32_t)rlps;
  LOG_LINE(g_logger, ss.str());
  return 1;
}


uint32_t BinEnc::encodeBin(uint32_t bin, const StaticCtx &ctxMdl, uint8_t ctxId, TensorType paramType)
{
    uint32_t rlps = ctxMdl.getRLPS(ctxId, paramType);
    uint32_t mps  = ctxMdl.getMPS(ctxId, paramType);

    std::ostringstream ss;
    ss << "===> Inside encodeBin: start with Range=" << m_Range << " Low=" << m_Low << " Bits left=" << m_BitsLeft;
    LOG_LINE(g_logger, ss.str());

    uint32_t rmps = m_Range - rlps;

    // determine if bin is LPS
    bool isLPS = (bin != mps);

    // update range
    m_Range = isLPS ? rlps : rmps;

    // update low if LPS
    if (isLPS)
        m_Low += rmps;

    // renormalize
    while (m_Range < 256)
    {
        m_Range <<= 1;
        m_Low   <<= 1;
        m_BitsLeft--;

        if (m_BitsLeft < 12)
            write_out();
    }

     ss .str("");
    ss << "==> encodeBin called: bin=" << bin << " ctxId=" << (int)ctxId 
       << " rlps=" << rlps << " mps=" << mps
       << " m_Range=" << m_Range << " m_Low=" << m_Low
       << " m_BitsLeft=" << m_BitsLeft;
    LOG_LINE(g_logger, ss.str());

    return 1;
}

uint32_t BinEnc::encodeBinEP( uint32_t bin )
{
  
    m_Low <<= 1;
    if (bin)
    {
        m_Low += m_Range;
    }
    m_BitsLeft--;
    if (m_BitsLeft < 12)
    {
      { 
        write_out();
      }
    }
    return 0;
}


uint32_t BinEnc::encodeBinsEP( uint32_t bins, uint32_t numBins )
{
    CHECK( bins >= ( 1u << numBins ), printf( "%i can not be coded with %i EP-Bins", bins, numBins ) )
  printf("Inside encodeBinsEP. Range=%d, low=%d, bits left=%d, bins=%d\n", m_Range, m_Low, m_BitsLeft, bins);
    std::ostringstream ss;
    ss << "===> Inside encodeBinsEP=" << m_Range << " Low=" << m_Low << " Bits left=" << m_BitsLeft << " bins=" << bins;
    LOG_LINE(g_logger, ss.str());

    if (m_Range == 256)
    {
        uint32_t remBins = numBins;
        while (remBins > 0)
        {
            uint32_t binsToCode = std::min<uint32_t>(remBins, 8); //code bytes if able to take advantage of the system's byte-write function
            uint32_t binMask    = (1 << binsToCode) - 1;
            uint32_t newBins    = (bins >> (remBins - binsToCode)) & binMask;
            m_Low               = (m_Low << binsToCode) + (newBins << 8); //range is known to be 256
            remBins            -= binsToCode;
            m_BitsLeft         -= binsToCode;
            if (m_BitsLeft < 12)
            {
                { 
                write_out();
                }
            }
        }
        printf("Inside encodeBinsEP. Range=%d, low=%d, bits left=%d, bins=%d\n", m_Range, m_Low, m_BitsLeft, bins);
        std::ostringstream ss;
        ss << "===> Inside encodeBinsEP=" << m_Range << " Low=" << m_Low << " Bits left=" << m_BitsLeft << " bins=" << bins;
        LOG_LINE(g_logger, ss.str());

        return 0;
    }
    while (numBins > 8)
    {
        numBins          -= 8;
        uint32_t pattern  = bins >> numBins;
        m_Low           <<= 8;
        m_Low            += m_Range * pattern;
        bins             -= pattern << numBins;
        m_BitsLeft       -= 8;
        if (m_BitsLeft < 12)
        {
            { 
            write_out();
            }
        }
    }
    m_Low     <<= numBins;
    m_Low      += m_Range * bins;
    m_BitsLeft -= numBins;
    if (m_BitsLeft < 12)
    { 
      {
        write_out();
        }
    }
    printf("Inside encodeBinsEP. Range=%d, low=%d, bits left=%d, bins=%d\n", m_Range, m_Low, m_BitsLeft, bins);
    
     ss .str("");
    ss << "===> Inside encodeBinsEP=" << m_Range << " Low=" << m_Low << " Bits left=" << m_BitsLeft << " bins=" << bins;
    LOG_LINE(g_logger, ss.str());

    return 0;
}


void BinEnc::write_out()
{
    uint32_t lead_byte = m_Low >> (24 - m_BitsLeft);
    m_BitsLeft += 8;
    m_Low &= 0xffffffffu >> m_BitsLeft;
    if (lead_byte == 0xff)
    {
        m_NumBufferedBytes++;
    }
    else
    {
        if (m_NumBufferedBytes > 0)
        {
            uint32_t carry      = lead_byte >> 8;
            uint8_t  byte       = m_BufferedByte + carry;
            m_BufferedByte       = lead_byte & 0xff;
            m_ByteBuf->push_back(byte);
            byte                = (0xff + carry) & 0xff;
            while (m_NumBufferedBytes > 1)
            {
                m_ByteBuf->push_back(byte);
                m_NumBufferedBytes--;
            }
        }
        else
        {
            m_NumBufferedBytes = 1;
            m_BufferedByte      = lead_byte;
        }
    }
}

void BinEnc::encodeBinTrm( unsigned bin )
{
  m_Range -= 2;
  if( bin )
  {
    m_Low      += m_Range;
    m_Low     <<= 7;
    m_Range     = 2 << 7;
    m_BitsLeft -= 7;
  }
  else if( m_Range >= 256 )
  {
    return;
  }
  else
  {
    m_Low     <<= 1;
    m_Range   <<= 1;
    m_BitsLeft--;
  }
  if( m_BitsLeft < 12 )
  {
    write_out();
  }
}

void BinEnc::finish()
{
  if( m_Low >> ( 32 - m_BitsLeft ) )
  {
    m_ByteBuf->push_back( m_BufferedByte + 1 );
    while( m_NumBufferedBytes > 1 )
    {
      m_ByteBuf->push_back( 0x00 );
      m_NumBufferedBytes--;
    }
    m_Low -= 1 << ( 32 - m_BitsLeft );
  }
  else
  {
    if( m_NumBufferedBytes > 0 )
    {
      m_ByteBuf->push_back( m_BufferedByte );
    }
    while( m_NumBufferedBytes > 1 )
    {
      m_ByteBuf->push_back( 0xff );
      m_NumBufferedBytes--;
    }
  }

  // add trailing 1
   m_Low >>= 8;
   m_Low <<= 1;
   m_Low++;
   m_BitsLeft--;
   // left align
   m_Low <<= (32 - (24-m_BitsLeft) );
   // write out starting from the leftmost byte
   for( unsigned i = 0; i < 24 - m_BitsLeft; i+=8 )
   {
     m_ByteBuf->push_back( (m_Low >> (24-i)) & 0xFF );
   }
}
