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
#include "CABACEncoder.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <sstream>
#include "Utils/global_logger.h"


void CABACEncoder::startCabacEncoding( std::vector<uint8_t>* pBytestream)
{
    m_BinEncoder.setByteStreamBuf(pBytestream);
    m_TensorMean = 0;
    m_BinEncoder.startBinEncoder();
}

void CABACEncoder::initCtxMdls(uint32_t numGtxFlags)
{
  TCABACEncoder<BinEnc>::xInitCtxModels(numGtxFlags);
}


uint64_t CABACEncoder::encodeTensorHeader(const int32_t* pWeights, uint32_t numWeights, const uint32_t* shape, uint32_t numDims, const std::string& tensor_name)
{

    uint64_t binsUsed = 0;
    // encode tensor type
    m_BinEncoder.encodeBinEP(static_cast<uint32_t>(m_tensorType)); // weight or bias
    binsUsed += 1; // 1 bit for tensor type
    // encode bitwidth
    m_BinEncoder.encodeBinsEP(static_cast<uint32_t>(m_tensorBitwidth), 3); // using 3 bits for bitwidth (up to 8 different bitwidths)
    binsUsed += 3; // 3 bits for bitwidth
    // encode number of dimensions
    m_BinEncoder.encodeBinsEP(numDims, 3); // using 4 bits for numDims (up to 8 dimensions)
    binsUsed += 3; // 3 bits for numDims
    uint16_t shapeBits = 0;

    // encode shape of each dimension
    for (uint32_t i = 0; i < numDims; i++)
    {
        uint32_t dimSize = shape[i];
        int bitlen = (dimSize == 0) ? 1 : 32 - __builtin_clz(dimSize);
        printf("CLZ results: %d\n", __builtin_clz(dimSize));
        m_BinEncoder.encodeBinsEP(bitlen-1, 5);
        //uae_v(5, bitlen - 1);
        m_BinEncoder.encodeBinsEP(dimSize, bitlen);
        //uae_v(bitlen, dimSize);
        shapeBits += 5 + bitlen; // bits used to encode this dimension
        printf("Encoded dimension %d: size=%d, bitlen=%d\n", i, dimSize, bitlen);
        binsUsed += 5 + bitlen;
    }

    int64_t sum = 0;
    int32_t max_abs = 0;
    uint32_t count = 0;

    const uint32_t chunkSize = 256 ; // small chunk for low RAM = for 32bits = 1KB
    uint32_t numChunks = (numWeights + chunkSize - 1) >> 8;;
    for (uint32_t chunk = 0; chunk < numChunks; chunk++)
    {
        for (uint32_t i = 0; i < chunkSize && (chunk * chunkSize + i) < numWeights; i++)
        {
            int32_t value = pWeights[chunk * chunkSize + i];
            sum += value;
            max_abs = std::max(max_abs, std::abs(value));
            count++;
        }
      }
      // Approximate mean using nearest power-of-2 shift
    uint32_t shift = std::ceil(std::log2(count));
    int32_t mean = sum >> shift; // integer shift division
    bool use_mean = (std::abs(mean) > 4); // threshold fixed at 4, can be tuned based on experiments
    m_useMean = use_mean;
    m_TensorMean = mean;
    m_BinEncoder.encodeBinEP(use_mean ? 1 : 0); // flag to indicate if mean is used
    binsUsed += 1;
    int bitwidth = getBitwidthFromEnum(m_tensorBitwidth);
    if (use_mean)
    {
      // 2. send/store mean
      printf("Encoding mean: %d with bitwidth: %d\n", mean, bitwidth);
      CABACEncoder::iae_v(bitwidth, mean); 
      binsUsed += bitwidth; // account for bits used to encode mean
      printf("Encoded mean: %d\n", mean);
    } else {
      m_TensorMean = 0; // if not using mean, set it to zero for encoding residuals
        printf("Mean not used, mean value: %d\n", mean);
    }


    g_logger->setTensorName(tensor_name);
    std::ostringstream ss;
    ss << "==> encodeTensorHeader: "
        << "tensor_name=" << tensor_name 
       << "tensorType=" << static_cast<uint32_t>(m_tensorType)
       << ", bitwidth=" << static_cast<uint32_t>(m_tensorBitwidth)
       << ", numDims=" << numDims
       << "use mean=" << use_mean
       << ", mean=" << mean
       << " overhead in bits=" << binsUsed; // rough estimate of header bits
    LOG_LINE(g_logger, ss.str());
    printf("==> encodeTensorHeader returning with binsUsed=%llu\n", binsUsed);
    return binsUsed;
}


void CABACEncoder::iae_v( uint8_t v, int32_t value )
{
  //PROFILE_SCOPE("iae_v", 0);
  printf("==> iae_v called with v=%d, value=%d\n", v, value);
    uint32_t pattern = uint32_t(value) & (uint32_t(0xFFFFFFFF) >> (32-v));
    printf("==> iae_v: pattern=0x%X\n", pattern);
    m_BinEncoder.encodeBinsEP( pattern, v );
}

void CABACEncoder::uae_v( uint8_t v, uint32_t value )
{
  //PROFILE_SCOPE("uae_v", 0);
    m_BinEncoder.encodeBinsEP( value, v );
}

void CABACEncoder::terminateCabacEncoding()
{
    //PROFILE_SCOPE("terminateCabacEncoding", 0);
    m_BinEncoder.encodeBinTrm(1);
    m_BinEncoder.finish();
}

void CABACEncoder::xEncRowSkip(uint8_t general_profile_idc, uint8_t rowSkipFlag,uint32_t layerWidth,uint32_t numWeights,int32_t* pChanZeroList, uint32_t codebook_size)
{
  //PROFILE_SCOPE("xEncRowSkip", 0);
  if(general_profile_idc == 1 && layerWidth > 1 && numWeights > layerWidth && codebook_size != 1)
  {
   
    m_BinEncoder.encodeBinEP( rowSkipFlag ? 1 : 0 );
    
    if(rowSkipFlag)
    {
      int32_t numRows = numWeights / layerWidth;
      for(int row = 0; row < numRows; row++)
      {
        // dummy  values
        //m_BinEncoder.encodeBin(pChanZeroList[row],m_CtxStore, 0, m_tensorType);
        
    }
    }
  }
}


uint64_t CABACEncoder::encodeWeights(const int32_t *pWeights, uint32_t numWeights)
{
  std::ostringstream ss;
  ss << "==> encodeWeights called with" 
     << "numWeights=" << numWeights;
  LOG_LINE(g_logger, ss.str());
  return xEncodeWeights<Trellis8States>( pWeights, numWeights);

}

TensorRole classifyTensor(const std::string& tensorName)
{
    std::string name = tensorName;
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);

    if (name.find("bias") != std::string::npos)
        return TensorRole::BIAS;
    if (name.find("norm") != std::string::npos || name.find("ln") != std::string::npos || name.find("bn") != std::string::npos)
        return TensorRole::WEIGHT_NORM;
    if (name.find("embedding") != std::string::npos || name.find("embed") != std::string::npos || name.find("position") != std::string::npos || name.find("pos") != std::string::npos || name.find("token") != std::string::npos)
        return TensorRole::WEIGHT_EMBEDDING;
    if (name.find("classifier") != std::string::npos || name.find("head") != std::string::npos)
        return TensorRole::WEIGHT_CLASSIFIER;

    if (name.find("attention") != std::string::npos || name.find("att") != std::string::npos)
    {
        if (name.find("q") != std::string::npos) return TensorRole::WEIGHT_ATT_Q;
        if (name.find("k") != std::string::npos) return TensorRole::WEIGHT_ATT_K;
        if (name.find("v") != std::string::npos) return TensorRole::WEIGHT_ATT_V;
        if (name.find("o") != std::string::npos) return TensorRole::WEIGHT_ATT_O;
        return TensorRole::OTHER;
    }

    if (name.find("conv") != std::string::npos) return TensorRole::WEIGHT_CONV;
    if (name.find("linear") != std::string::npos || name.find("fc") != std::string::npos) return TensorRole::WEIGHT_FC;

    return TensorRole::UNKNOWN;
    printf("Classified tensor '%s' as role %d\n", tensorName.c_str(), static_cast<int>(classifyTensor(tensorName)));
}

bool decideMeanFlag (const std::string& tensorName){
    TensorRole role = classifyTensor(tensorName);
    // For simplicity, we can decide to use local mean for all weights except biases and LayerNorm parameters, which often have small values and may not benefit from mean-based residuals.
    bool meanFlag = 0;
    switch(role){
      case TensorRole::WEIGHT_CONV:
      case TensorRole::WEIGHT_ATT_Q:
      case TensorRole::WEIGHT_ATT_K:
      case TensorRole::WEIGHT_ATT_V:
      case TensorRole::WEIGHT_ATT_O:
      case TensorRole::WEIGHT_CLASSIFIER:
      case TensorRole::WEIGHT_EMBEDDING:
           meanFlag = false;    // small variance
          break;
      case TensorRole::WEIGHT_FC:
      case TensorRole::WEIGHT_NORM:
      case TensorRole::BIAS:
          meanFlag = true;    // large variance
          break;
      default:
          meanFlag = true;    // conservative fallback
          break;
      }
}





