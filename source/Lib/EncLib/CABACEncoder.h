/* -----------------------------------------------------------------------------
The copyright in this software is being made available under the Clear BSD
License, included below. No patent rights, trademark rights and/or
other Intellectual Property Rights other than the copyrights concerning
the Software are granted under this license.

The Clear BSD License

Copyright (c) 2019-2025, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The NNCodec Authors.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted (subject to the limitations in the disclaimer below) provided that
the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

     * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.


------------------------------------------------------------------------------------------- */
#ifndef __CABACENC__
#define __CABACENC__

#include "../CommonLib/ContextModel.h"
#include "../CommonLib/ContextModeler.h"
#include "../CommonLib/Quant.h"
#include "../CommonLib/Scan.h"
#include "BinEncoder_simple.h"
#include <bitset>
#include <limits>
#include <iostream>
#include <algorithm>
#include "Utils/global_logger.h" 
#include <sstream>
#include <cstdint>
#include <cmath>

template< typename TBinEnc >
class TCABACEncoder
{
protected:
  __inline void xInitCtxModels(uint32_t numGtxFlags)
  {
    m_NumGtxFlags = numGtxFlags;
    m_CtxModeler.init(numGtxFlags);
  }

  template< uint32_t (TBinEnc::*FuncBinEnc)(uint32_t, const StaticCtx&, uint8_t, TensorType) >
  __inline uint32_t xEncRemAbs( int32_t value, uint16_t k)
  {
    //printf("==> EncWeight: xEncRemAbs value=%d\n", value);
    uint32_t scaledBits           = 0;
    uint8_t minusBits = 0;

    uint32_t bitwidth = getBitwidthFromEnum(m_tensorBitwidth);

    if (bitwidth < 2){
      return m_BinEncoder.encodeBinsEP( value, bitwidth );
    }

    //extract MSBs 
    uint32_t msb1 = (value >> (bitwidth - 1)) & 0x1;
    uint32_t msb2 = (value >> (bitwidth - 2)) & 0x1;

    // encode MSBs
    scaledBits += (m_BinEncoder.*FuncBinEnc)( msb1, m_CtxStore, 6, m_tensorType );
    scaledBits += (m_BinEncoder.*FuncBinEnc)( msb2, m_CtxStore, 7, m_tensorType );
    minusBits += 2;

    uint32_t msb3 = 0, msb4 = 0, msb5 =0, msb6 =0;

    if (m_tensorBitwidth == TensorBitwidth::BW_12) {
      msb3 = (value >> (bitwidth - 3)) & 0x1;
      scaledBits += (m_BinEncoder.*FuncBinEnc)( msb3, m_CtxStore, 8, m_tensorType );
      msb4 = (value >> (bitwidth - 4)) & 0x1;
      scaledBits += (m_BinEncoder.*FuncBinEnc)( msb4, m_CtxStore, 9, m_tensorType );
      minusBits += 2;
    } else if (m_tensorBitwidth >= TensorBitwidth::BW_16) {
      msb3 = (value >> (bitwidth - 3)) & 0x1;
      scaledBits += (m_BinEncoder.*FuncBinEnc)( msb3, m_CtxStore, 8, m_tensorType );
      msb4 = (value >> (bitwidth - 4)) & 0x1;
      scaledBits += (m_BinEncoder.*FuncBinEnc)( msb4, m_CtxStore, 9, m_tensorType );
      msb5 = (value >> (bitwidth - 5)) & 0x1;
      scaledBits += (m_BinEncoder.*FuncBinEnc)( msb5, m_CtxStore, 10, m_tensorType );
      msb6 = (value >> (bitwidth - 6)) & 0x1;
      scaledBits += (m_BinEncoder.*FuncBinEnc)( msb6, m_CtxStore, 11, m_tensorType );
      minusBits += 4;
    }

    uint32_t baseMask = (1 << (bitwidth - minusBits)) - 1;
    uint32_t value_no_msb = value & baseMask;

    uint8_t maxUnary = 15; // to prevent infinite loops, can be set according to expected value range
    uint8_t k_upd = k+1;
    uint32_t q = value_no_msb >> k_upd;
    uint32_t r = value_no_msb & ((1 << k_upd) - 1);

    uint32_t remainingBits = bitwidth - (minusBits + k_upd); // 2 bits for MSBs and k bits for q

   // if (q > maxUnary)
   // {
   //     scaledBits += m_BinEncoder.encodeBinsEP(value & ((1 << remainingBits) - 1), remainingBits ); // encode all remaining bits as EP bins if q exceeds maxUnary
   // } else{
        // unary prefix
        for(uint32_t i = 0; i < q; i++){
          m_BinEncoder.encodeBinEP(1);
          scaledBits += 1;
        }


        m_BinEncoder.encodeBinEP(0);
        scaledBits += 1;

         // encode suffix

        // suffix
        m_BinEncoder.encodeBinsEP(r, k_upd);
        scaledBits += k_upd;
  //  }
 
    std::ostringstream ss;
    ss << "==> xEncRemAbs: value=" << value
       << ", msb1=" << msb1
       << ", msb2=" << msb2
       << ", msb3=" << msb3 
       << ", msb4=" << msb4
       << ", msb5=" << msb5
       << ", msb6=" << msb6
       << ", q=" << q // example q calculation for logging
       << ", r=" << r // example r calculation for logging
       << ", kupd=" << (int)k_upd
       << " minusBits=" << (int)minusBits
       << " scaledBits=" << scaledBits;
    LOG_LINE(g_logger, ss.str());
    return scaledBits;
  }

  template< uint32_t (TBinEnc::*FuncBinEnc)(uint32_t, const StaticCtx&, uint8_t, TensorType) >
  __inline uint32_t xEncWeight( int32_t value, uint8_t k)
  {
    uint32_t sigFlag        = value != 0 ? 1 : 0;
    int32_t  sigctx         = m_CtxModeler.getSigCtxId( );
    uint32_t scaledBits     = (m_BinEncoder.*FuncBinEnc)(sigFlag, m_CtxStore, sigctx, m_tensorType);
    
    if (sigFlag)
    {
      uint32_t signFlag = value < 0 ? 1 : 0;
      int32_t signCtx;

      signCtx = m_CtxModeler.getSignFlagCtxId();
      scaledBits += (m_BinEncoder.*FuncBinEnc)(signFlag, m_CtxStore, signCtx, m_tensorType);     

      uint32_t remAbsLevel = abs(value) - 1;

      if (abs(value) > 5){
        // bypass gtx flags and directly encode remAbsLevel using xEncRemAbs
        scaledBits += (m_BinEncoder.*FuncBinEnc)(1, m_CtxStore, 12, m_tensorType); // set branch flag to 1 to indicate large residual
        remAbsLevel -= 5; // we can subtract 5 here because values <=5 are handled in the small branch, this way we encode a smaller number in xEncRemAbs which is more efficient
        scaledBits += xEncRemAbs<FuncBinEnc>( remAbsLevel, k); 
        std::ostringstream ss;
        ss << "==> xEncWeight (large branch): value=" << value
          << ", sigFlag=" << sigFlag
          << ", signFlag=" << signFlag
          << ", k=" << (int)k
          << ", remAbsLevel=" << remAbsLevel
          << "m_tensorType=" << (int)m_tensorType;
        LOG_LINE(g_logger, ss.str()); 
      } else {
        scaledBits += (m_BinEncoder.*FuncBinEnc)(0, m_CtxStore, 12, m_tensorType); // set branch flag to 0 to indicate small residual

        uint32_t grXFlag = remAbsLevel ? 1 : 0; //greater1
        int32_t ctxIdx;

        ctxIdx = m_CtxModeler.getGtxCtxId( signFlag);
        scaledBits += (m_BinEncoder.*FuncBinEnc)(grXFlag, m_CtxStore, ctxIdx, m_tensorType);

        uint32_t numGreaterFlagsCoded = 1;
        //printf("==> EncWeight: signctx=%d, ctxidx=%d, signFlag=%d, grXFlag=%d, scaledBits=%d\n", signCtx, ctxIdx, signFlag, grXFlag, scaledBits);
        while (grXFlag && (numGreaterFlagsCoded < m_NumGtxFlags) )
        {
          remAbsLevel--;
          grXFlag = remAbsLevel ? 1 : 0;
          ctxIdx =  m_CtxModeler.getGtxCtxId(signFlag);         
          scaledBits += (m_BinEncoder.*FuncBinEnc)(grXFlag, m_CtxStore, ctxIdx, m_tensorType);        
          numGreaterFlagsCoded++;
          //printf("==> EncWeight: numGreaterFlagsCoded=%d, ctxidx=%d, remAbsLevel=%d, grXFlag=%d, scaledBits=%d\n", numGreaterFlagsCoded, ctxIdx, remAbsLevel, grXFlag, scaledBits);
        }

        std::ostringstream ss;
        ss << "==> xEncWeight (small branch): value=" << value
          << ", sigFlag=" << sigFlag
          << ", signFlag=" << signFlag
          << ", grXFlag=" << grXFlag
          << ", k=" << (int)k
          << ", remAbsLevel=" << remAbsLevel
          << "m_tensorType=" << (int)m_tensorType;
        LOG_LINE(g_logger, ss.str());
      }
      
      
    }
    return scaledBits;
  }

protected:
  StaticCtx            m_CtxStore;
  ContextModeler       m_CtxModeler;
  TBinEnc              m_BinEncoder;
  uint32_t             m_NumGtxFlags;
  TensorBitwidth       m_tensorBitwidth;
  TensorType           m_tensorType;
  int32_t              m_TensorMean;
  bool                 m_useMean;



};


class CABACEncoder : protected TCABACEncoder<BinEnc>
{
public:
  CABACEncoder() {}
  ~CABACEncoder() {}

  void      startCabacEncoding      (std::vector<uint8_t>* pBytestream );
  void      initCtxMdls             (uint32_t numGtxFlags);

  void      terminateCabacEncoding  ();
  void      iae_v                   (uint8_t v,int32_t value);
  void      uae_v                   (uint8_t v,uint32_t value);

  uint64_t     encodeTensorHeader       ( const int32_t* pWeights, uint32_t numWeights, const uint32_t* shape, uint32_t numDims, const std::string& tensor_name = "");

  void      encodeWeightDirect(int32_t weight, uint16_t k = 2) { return encodeWeightVal(weight, k); }

  uint64_t   encodeWeights(const int32_t* pWeights, uint32_t numWeights);

  void      setBitwidthAndType(TensorBitwidth bitwidth, TensorType type) {
      m_tensorBitwidth = bitwidth;
      m_tensorType = type;
  }

private:

  __inline void encodeWeightVal(int32_t weightInt, uint16_t k = 2)
  {
    this->TCABACEncoder<BinEnc>::template
    xEncWeight<&BinEnc::encodeBin>(weightInt, k);

  }

template <class trellisDef >
uint64_t EncodeWeightsBase( int32_t* pWeights, uint32_t numWeights)
  {
    { printf("==> encodeTensorbase: numWeights=%d\n", numWeights);

    uint64_t scaledBits = 0;
    int32_t localMean = 0;
    int shift = 0;

    bool useDelta = false;

    const uint32_t chunkSize = 256 ; // small chunk for low RAM = for 32bits = 1KB 
    uint32_t numChunks = (numWeights + chunkSize - 1) >> 8;;

    for (uint32_t c = 0; c < numChunks; c++)
    {
        uint32_t start = c * chunkSize;
        uint32_t end   = std::min(start + chunkSize, numWeights);
        uint32_t len   = end - start;

        printf("Processing chunk %d/%d: start=%d, end=%d, len=%d\n", c+1, numChunks, start, end, len);

        localMean = m_TensorMean;

        // Step 2: find mean absolute value
        int64_t sumRes = 0;
        int32_t residual = 0;
        for (uint32_t i = start; i < end; i++)
        {
            residual = pWeights[i] - localMean;
            sumRes += std::abs(residual);
        }

        int32_t meanRes = sumRes >> 8; // meanRes = sumRes / len, using shift for efficiency (len is 256)
        if (meanRes == 0) meanRes = 1; // to avoid division by zero
        printf("Chunk %d: localMean=%d, meanRes=%d\n", c, localMean, meanRes);

        // compute k per chunk
        int k = 0;
        if      (meanRes < 8)       k = 0;
        else if (meanRes < 32)      k = 1;
        else if (meanRes < 256)     k = 2;
        else if (meanRes < 1024)    k = 3;
        else                        k = 3;

        CABACEncoder::uae_v(2, k); // send k as 2-bit fixed length for simplicity
        scaledBits += 2; // account for bits used to encode k

        // Step 3:  scale bt power of two using shifts 
        uint8_t shift = getShiftFromMeanAndK(m_tensorBitwidth, m_TensorMean, k);
      
        int32_t scaled = 0;
        for (uint32_t i = start; i < end; i++)
        {
            residual = pWeights[i] - localMean;
            
            if (shift > 0)
                scaled = (residual + (residual >= 0 ? (1 << (shift-1)) : (1 << (shift-1)))) >> shift;
            else
                scaled = residual;


            scaledBits += xEncWeight<&BinEnc::encodeBin>(scaled, k);
            m_CtxModeler.updateNeighborCtx(scaled);  
            std::ostringstream ss;
            ss << "==> EncWeight: weight=" << pWeights[i]
               << ", residual=" << residual
               << ", scaled=" << scaled
               << ", localMean=" << localMean
               << ", shift=" << (int)shift
               << ", k=" << k
               << " m_tensorType=" << (int)m_tensorType
               << " scaledBits=" << scaledBits;
               LOG_LINE(g_logger, ss.str());

        }

       std::ostringstream ss;
        ss << "==> EncodeWeightsBase: numWeights=" << numWeights
       << ", localMean=" << localMean
          << ", k=" << k
       << ", shift=" << (int)shift
       << " chunk " << c << "/" << numChunks
       << " chunksize=" << len
       << " scaledBits=" << scaledBits
       ;
      LOG_LINE(g_logger, ss.str());
    }

    return scaledBits;
  }
}

  template <class trellisDef >
  uint64_t xEncodeWeights(const int32_t* pWeights, uint32_t numWeights)
  {
    uint64_t bits;

    m_CtxModeler.resetNeighborCtx();

    bits = EncodeWeightsBase<trellisDef>(pWeights, numWeights);
    return bits;
  }


  void xEncRowSkip     ( uint8_t general_profile_idc, uint8_t rowSkipFlag,uint32_t layerWidth,uint32_t numWeights,int32_t* pChanZeroList, uint32_t codebook_size);
  uint8_t                       m_ParamOptFlag;

};



#endif // !__CABACENCIF__