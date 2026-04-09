
#include <random>
#include <algorithm>
#include <iostream>
#include "BinDecoder.h"
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

    // Primes the 15-bit Value window from first 2 bytes — unchanged
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

    return bin;
}

uint32_t BinDec::decodeBin(StaticCtx &ctxMdl, uint8_t ctxId, TensorType paramType)
{
    //printf("DecodeBin called: range %d value %d \n", m_Range, m_Value );
    uint32_t rlps = ctxMdl.getRLPS(ctxId, paramType); OP_MEM();
    uint32_t mps  = ctxMdl.getMPS(ctxId, paramType); OP_MEM();

    //uint32_t rmps = m_Range - rlps;
    uint32_t rmps = OP_SUB(m_Range - rlps);

    // determine LPS
    bool isLPS = OP_CMP(m_Value >= OP_SHL(rmps, 7)); OP_BRANCH();

    // reconstruct bin
    uint32_t bin = isLPS ? (mps ^ 1) : mps; OP_BRANCH();

    // update range 
    m_Range = isLPS ? rlps : rmps; OP_BRANCH();

    // update value if LPS
    if (isLPS) {
        OP_BRANCH();
        m_Value = OP_SUB(m_Value - OP_SHL(rmps, 7));
    }

       
    uint32_t n = OP_SUB(clz32(m_Range) - 23);

    m_Range = OP_SHL(m_Range, n);
    m_Value = OP_SHL(m_Value, n);
    m_BitsNeeded = OP_ADD(m_BitsNeeded + n);

    if (OP_CMP(m_BitsNeeded >= 0)) {
        OP_BRANCH();

        uint8_t byte = *m_ByteStreamPtr++;
        m_Value = OP_ADD(m_Value + OP_SHL(byte, m_BitsNeeded));
        OP_MEM();

        m_BitsNeeded = OP_SUB(m_BitsNeeded - 8);
        m_BytesRead++; // optional: OP_MEM()
    }

    g_ops.regularBins++;

    return bin;
}

uint32_t BinDec::decodeBinEP()
{
    m_Value = OP_ADD(m_Value + m_Value);
    ++m_BitsNeeded;
    if (OP_CMP(m_BitsNeeded >= 0)) {
        OP_BRANCH();
        uint8_t byte = *m_ByteStreamPtr++;
        m_Value = OP_ADD(m_Value + byte); OP_MEM();
        m_BitsNeeded = -8;
    }
    uint32_t bin = 0;
    uint32_t SR = OP_SHL(m_Range, 7);

    if (OP_CMP(m_Value >= SR)) {
        OP_BRANCH();

        m_Value = OP_SUB(m_Value - SR);
        bin = 1;
    }
    g_ops.bypassBins++;
    return bin;
}

uint32_t BinDec::decodeBinsEP( uint32_t numBins )
{
    if (OP_CMP(m_Range == 256)) {
        OP_BRANCH();
        uint32_t remBins = numBins;
        uint32_t bins    = 0;
        while (OP_CMP(remBins > 0)) {
            g_ops.loops++;
            OP_BRANCH();

            uint32_t binsToRead = std::min<uint32_t>(remBins, 8);
            uint32_t binMask    = OP_SUB((1 << binsToRead) - 1);
           uint32_t newBins = OP_SHR(m_Value, (15 - binsToRead)) & binMask;

            bins = OP_ADD((bins << binsToRead) | newBins);
            m_Value = OP_SHL(m_Value, binsToRead) & 0x7FFF;
            remBins            -= binsToRead;
            m_BitsNeeded       += binsToRead;
           if (OP_CMP(m_BitsNeeded >= 0)) {
                OP_BRANCH();
                uint8_t byte = *m_ByteStreamPtr++;
                m_Value |= OP_SHL(byte, m_BitsNeeded);
                OP_MEM();

                m_BitsNeeded = OP_SUB(m_BitsNeeded - 8);
            }
        }
        g_ops.bypassBins += numBins;
        return bins;
    }
    uint32_t remBins = numBins;
    uint32_t bins = 0;
    while (OP_CMP(remBins > 8)) {
        g_ops.loops++;
        OP_BRANCH();
        uint8_t byte = *m_ByteStreamPtr++; OP_MEM();

        uint32_t tmp1 = OP_SHL(m_Value, 8);
        uint32_t tmp2 = OP_SHL(byte, 8 + m_BitsNeeded);
        m_Value = OP_ADD(tmp1 + tmp2);
        uint32_t SR = OP_SHL(m_Range, 15);

        for (uint32_t i = 0; OP_CMP(i < 8); i++) {
            g_ops.loops++;
            OP_BRANCH();
            bins = OP_ADD(bins + bins);
            SR = OP_SHR(SR, 1);
            if (OP_CMP(m_Value >= SR)) {
                OP_BRANCH();
                bins = OP_ADD(bins + 1);
                m_Value = OP_SUB(m_Value - SR);
            }
        }
        remBins = OP_SUB(remBins - 8);
        m_BytesRead++;
    }

    m_BitsNeeded = OP_ADD(m_BitsNeeded + remBins);
    m_Value = OP_SHL(m_Value, remBins);
    if (OP_CMP(m_BitsNeeded >= 0)) {
        OP_BRANCH();
        uint8_t byte = *m_ByteStreamPtr++; OP_MEM();
        uint32_t tmp1 = OP_SHL(byte, m_BitsNeeded);
        m_Value = OP_ADD(m_Value + tmp1);
        m_BitsNeeded = OP_SUB(m_BitsNeeded - 8);
        m_BytesRead++;
    }

    uint32_t SR = OP_SHL(m_Range, remBins + 7);
    for (uint32_t i = 0; OP_CMP(i < remBins); i++) {
        g_ops.loops++;
        OP_BRANCH();
        bins = OP_ADD(bins + bins);
        SR = OP_SHR(SR, 1);
        if (OP_CMP(m_Value >= SR)) {
            OP_BRANCH();
            bins = OP_ADD(bins + 1);
            m_Value = OP_SUB(m_Value - SR);
        }
    }

    g_ops.bypassBins += numBins;
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

