#include "CABACDecoder.h"
#include <iostream>
#include <sstream>

void BACDecoder::startBacDecoding(uint8_t* pBytestream)
{
  //g_logger->setTensorName("CABACDecoder_log");
  m_BinDecoder.setByteStreamBuf(pBytestream);
  m_BinDecoder.startBinDecoder();
  //printf("CABACDecoder: Started decoding\n");
}

void BACDecoder::initCtxModels(uint32_t cabac_unary_length)
{
  m_NumGtxFlags = cabac_unary_length;
  m_CtxModeler.init(cabac_unary_length);
  //printf("CABACDecoder: Context models initialized with cabac_unary_length=%d\n", cabac_unary_length);
}


int32_t BACDecoder::iae_v(uint8_t v)
{
  uint32_t pattern = m_BinDecoder.decodeBinsEP(v);
  return int32_t(pattern << (32 - v)) >> (32 - v);
}

uint32_t BACDecoder::uae_v(uint8_t v)
{
  return m_BinDecoder.decodeBinsEP(v);
}


uint64_t BACDecoder::decodeTensorHeader(uint32_t* shape, uint32_t& numDims, TensorMeta &tensor)
{
  uint64_t binsRead = 0;
 // //printf("==> decodeTensorHeader called\n");
  // decode tensor id
  tensor.tensorId = m_BinDecoder.decodeBinsEP(10);
  // decode tensor type
  m_tensorType = static_cast<TensorType>(m_BinDecoder.decodeBinEP());
  tensor.tensorType = m_tensorType;
//  //printf("Decoded tensor type: %d\n", static_cast<uint32_t>(m_tensorType));
  binsRead += 1; // 1 bit for tensor type

  // decode bitwidth
  m_tensorBitwidth = static_cast<TensorBitwidth>(m_BinDecoder.decodeBinsEP(3));
  tensor.tensorBitwidth = m_tensorBitwidth;
  int bitwidth = getBitwidthFromEnum(m_tensorBitwidth);
 // //printf("Decoded tensor bitwidth: %d\n", static_cast<uint32_t>(m_tensorBitwidth));
  binsRead += 3; // 3 bits for bitwidth

  // decode number of dimensions
  numDims = m_BinDecoder.decodeBinsEP(3);
  tensor.numDims = numDims;
 // //printf("Decoded number of dimensions: %d\n", numDims);
  binsRead += 3; // 3 bits for numDims

  // decode shape of each dimension
  uint32_t bitlenMinus1 = 0, bitlen = 0;

  for (uint32_t i = 0; i < numDims; i++)
  {
    //bitlenMinus1 = uae_v(5);
    bitlenMinus1 = m_BinDecoder.decodeBinsEP(5);
    bitlen = bitlenMinus1 + 1;
   // //printf("Decoded bitlen for dimension %d: %d, -1:%d\n", i, bitlen, bitlenMinus1);
    //shape[i] = uae_v(bitlen);
    shape[i] = m_BinDecoder.decodeBinsEP(bitlen);
  //  //printf("Decoded dimension %d size: %d\n", i, shape[i]);
    binsRead += 5 + bitlen; // bits used to decode this dimension
  }

  m_useMean = m_BinDecoder.decodeBinEP(); // read flag for mean usage
  binsRead += 1; // 1 bit for mean usage flag

  if (m_useMean)
    {
      m_TensorMean = iae_v(bitwidth); // read mean value
      binsRead += bitwidth; // account for bits used to decode mean
   //   //printf("Decoded mean: %d\n", m_TensorMean);
    } else {
      m_TensorMean = 0; // if mean not used, set it to zero for decoding residuals
    //  //printf("Mean not used, mean value set to: %d\n", m_TensorMean);
    }

  return binsRead;
}


uint64_t BACDecoder::decodeWeightsChunks(int32_t* pWeights , uint32_t numWeights)
{

  int width = getBitwidthFromEnum(m_tensorBitwidth);
  uint64_t scaledBits = 0;

  const uint32_t chunkSize = 2048 ; // small chunk for low RAM = for 32bits = 8KB
  uint32_t numChunks = (numWeights + chunkSize - 1) >> 11; // 11 because of 2048

  for (uint32_t c = 0; c < numChunks; c++)
  {
    m_CtxModeler.resetNeighborCtx();

    uint32_t start = c * chunkSize;
    uint32_t end   = std::min(start + chunkSize, numWeights);

    // ------------ read chunk skip flag ----------------
    bool skipChunk = m_BinDecoder.decodeBinEP();
    scaledBits += 1;

    if (skipChunk){ /// raw ep bins decoding
      for (uint32_t i = start; i < end; i++){
        pWeights[i] = iae_v(width);
        scaledBits += width;
      }
      //printf ("Tensor decoded as raw EP bins.\n");
      continue;
    }

    // ---------- read k parameter for Rice-Golomb coding
    uint8_t k = uae_v(2); // read k as 2-bit fixed length for simplicity
    scaledBits += 2;

    // read shift for scaling
    uint8_t shift = getShiftFromMeanAndK(m_tensorBitwidth, m_TensorMean, k);
   // //printf("Calculated shift for chunk %d: %d\n", c, shift);

    // --------------- BAC decode weights
    for (uint32_t i = start; i < end; i++)
    {
      int32_t decodedVal = 0;
      scaledBits += decodeWeightVal(decodedVal, k); 
      int32_t residual = shift > 0 ? (decodedVal << shift) : decodedVal;

      pWeights[i] =  residual + m_TensorMean;
     // //printf("Decoded weight %d: value=%d\n", i,  pWeights[i]);
      m_CtxModeler.updateNeighborCtx(decodedVal);

    }
  }
  return scaledBits;
}


uint64_t BACDecoder::decodeWeights(int32_t *pWeights, uint32_t numWeights)
{

  return decodeWeightsChunks(pWeights, numWeights); 

}


uint64_t BACDecoder::decodeWeightVal(int32_t &decodedIntVal, uint8_t k )
{ 
  uint64_t bitsUsed = 0;

  const int32_t sigctx = m_CtxModeler.getSigCtxId();
  uint32_t sigFlag = m_BinDecoder.decodeBin(m_CtxStore, sigctx, m_tensorType);
 // //printf("Decoded sigFlag: %d\n", sigFlag);
  bitsUsed += 1; // 1 bit for sigFlag

  decodedIntVal = 0;

  if (!sigFlag)
    return bitsUsed;
  

  // sign 
  int32_t signCtx = m_CtxModeler.getSignFlagCtxId();
  uint32_t signFlag = m_BinDecoder.decodeBin(m_CtxStore, signCtx, m_tensorType);
//  //printf("Decoded signFlag: %d\n", signFlag);
  bitsUsed += 1; // 1 bit for signFlag

  // branch flag
  uint32_t branchFlag = m_BinDecoder.decodeBin(m_CtxStore, 12, m_tensorType); // assuming context 8 is for branch flag
  bitsUsed += 1; // 1 bit for branch flag

  if (branchFlag)
  {
    // large residual case, directly decode remAbsLevel without gtx flags
    uint32_t remAbsLevel = 0;
    bitsUsed += decodeAbsRem(remAbsLevel, k);
    decodedIntVal = signFlag ? -int32_t(remAbsLevel + 6) : int32_t(remAbsLevel + 6);
    
    return bitsUsed;
  } else {
    // small residual case, decode gtx flags first
    uint32_t remAbsLevel = 0; 
    uint32_t grXFlag = 0;
    uint8_t numGreaterFlagsDecoded = 0;

    do {
      uint32_t ctxIdx = m_CtxModeler.getGtxCtxId(signFlag);
      grXFlag = m_BinDecoder.decodeBin(m_CtxStore, ctxIdx, m_tensorType);
      bitsUsed  += 1; // 1 bit for grXFlag
      if (grXFlag)
        remAbsLevel++;
      numGreaterFlagsDecoded++;

      ////printf("Decoded grXFlag: %d (numGreaterFlagsDecoded=%d)\n", grXFlag, numGreaterFlagsDecoded);
    } while (grXFlag && numGreaterFlagsDecoded < m_NumGtxFlags);

    if (grXFlag) { // last grxFlag means decoded value greater than four
      remAbsLevel ++;
    }

    decodedIntVal = remAbsLevel + 1; // add 1 to get the original abs value
    decodedIntVal = signFlag ? -decodedIntVal : decodedIntVal;

  // //printf("Decoded weight value: %d\n", decodedIntVal);

    return bitsUsed;
  }
}

int32_t BACDecoder::decodeAbsRem(uint32_t& remainder, uint32_t k)
{
  uint32_t binsUsed = 0;
  uint32_t bitwidth = getBitwidthFromEnum(m_tensorBitwidth);
  uint8_t plusBits = 0;

  if (bitwidth < 2)
  {
      remainder = m_BinDecoder.decodeBinsEP(bitwidth);
      return bitwidth;
  }

  // ---- 1. Decode MSBs (context-coded) ----
  uint32_t msb1 = m_BinDecoder.decodeBin(m_CtxStore, 6, m_tensorType);
  uint32_t msb2 = m_BinDecoder.decodeBin(m_CtxStore, 7, m_tensorType);
  binsUsed += 2; // 2 bits for MSBs
  plusBits += 2;

  uint32_t msb3 = 0, msb4 = 0, msb5 = 0, msb6 = 0;
  if (m_tensorBitwidth == TensorBitwidth::BW_12) {
    msb3 = m_BinDecoder.decodeBin(m_CtxStore, 8, m_tensorType);
    msb4 = m_BinDecoder.decodeBin(m_CtxStore, 9, m_tensorType);
    binsUsed += 2; // 2 bits for MSBs
    plusBits += 2;
  } else if (m_tensorBitwidth >= TensorBitwidth::BW_16) {
      msb3 = m_BinDecoder.decodeBin(m_CtxStore, 8, m_tensorType);
      msb4 = m_BinDecoder.decodeBin(m_CtxStore, 9, m_tensorType);
      msb5 = m_BinDecoder.decodeBin(m_CtxStore, 10, m_tensorType);
      msb6 = m_BinDecoder.decodeBin(m_CtxStore, 11, m_tensorType);
      binsUsed += 4; // 4 bits for MSBs
      plusBits += 4;
  }

  // uint32_t value = 0;
  // value |= (msb1 << (bitwidth - 1));
  // value |= (msb2 << (bitwidth - 2));

  // ---- 2. Decode unary prefix ----
  uint32_t q = 0;
  uint8_t k_upd = k + 1;

  //uint32_t maxQ = ((1u << (bitwidth - plusBits)) - 1) >> k_upd;

  while (true)
  {
      uint32_t bin = m_BinDecoder.decodeBinEP();
      binsUsed++;
      if (bin == 0)
          break;
      q++;
      //if (q >= maxQ)
          //break;
  }

  // ---- 3. Decode suffix ----
  uint32_t r = m_BinDecoder.decodeBinsEP(k_upd);
  binsUsed += k_upd; // k_upd bits for suffix


  // ---- 4. Reconstruct value from MSBs, unary prefix, and suffix ----
  uint32_t lowerMask = (1u << (bitwidth - plusBits)) - 1;
  uint32_t lower = (q << k_upd) | r;
  lower &= lowerMask;

    uint32_t value = (msb1 << (bitwidth - 1)) |
                     (msb2 << (bitwidth - 2)) |
                     (msb3 << (bitwidth - 3)) |
                     (msb4 << (bitwidth - 4)) |
                     (msb5 << (bitwidth - 5)) |
                     (msb6 << (bitwidth - 6)) |
                     lower;

  remainder = value;


  return binsUsed;
}

uint32_t BACDecoder::getBytesRead()
{
  return m_BinDecoder.getBytesRead();
}

uint32_t BACDecoder::terminateBacDecoding()
{
  if (m_BinDecoder.decodeBinTrm())
  {
    m_BinDecoder.finish();
    return m_BinDecoder.getBytesRead();
  }
  CHECK(1, "Terminating Bin not received!");
}
