#include "CABACDecoder.h"

#include <iostream>
#include <sstream>

static constexpr uint32_t MAX_TENSORS_BITS = 12;   // allows up to 4096 tensors

void BACDecoder::startBacDecoding(uint8_t* pBytestream)
{
  //g_logger->setTensorName("CABACDecoder_log");
  m_BinDecoder.setByteStreamBuf(pBytestream);
  // m_BinDecoder.startBinDecoder();
  ////printf("CABACDecoder: Started decoding\n");
}

void BACDecoder::initCtxModels(uint32_t cabac_unary_length)
{
  m_NumGtxFlags = cabac_unary_length;
  m_CtxModeler.init(cabac_unary_length);
  ////printf("CABACDecoder: Context models initialized with cabac_unary_length=%d\n", cabac_unary_length);
}

uint8_t* BACDecoder::getBytestreamPtr()
{
  return m_BinDecoder.getByteStreamPtr();
}

void BACDecoder::setByteStreamPtr(uint8_t* ptr)
{
  m_BinDecoder.setByteStreamPtr(ptr);
}
void BACDecoder::setBytesRead(uint32_t bytesRead)
{
  m_BinDecoder.setBytesRead(bytesRead);
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


uint64_t BACDecoder::decodeTensorHeader(uint32_t* shape, uint32_t& numDims, TensorMeta &tensor, BitstreamReader& reader)
{
  uint64_t binsRead = 0;

  // decode tensor id
  tensor.tensorId = reader.readBits(MAX_TENSORS_BITS); // read numTensors
  //tensor.tensorId = m_BinDecoder.decodeBinsEP(MAX_TENSORS_BITS);
  binsRead += 12; 
  // decode tensor type
  m_tensorType = static_cast<TensorType>(reader.readBits(1)); // weight or bias
  // m_tensorType = static_cast<TensorType>(m_BinDecoder.decodeBinEP());
  tensor.tensorType = m_tensorType;
  // printf("Decoded tensor type: %d\n", static_cast<uint32_t>(m_tensorType));
  binsRead += 1; // 1 bit for tensor type

  // decode bitwidth
  m_tensorBitwidth = static_cast<TensorBitwidth>(reader.readBits(3)); // using 3 bits for bitwidth (up to 8 different bitwidths)
  // m_tensorBitwidth = static_cast<TensorBitwidth>(m_BinDecoder.decodeBinsEP(3));
  tensor.tensorBitwidth = m_tensorBitwidth;
 // //printf("Decoded tensor bitwidth: %d\n", static_cast<uint32_t>(m_tensorBitwidth));
  binsRead += 3; // 3 bits for bitwidth

  // decode number of dimensions
  numDims = reader.readBits(3); // using 3 bits for numDims (up to 8 dimensions)
  // numDims = m_BinDecoder.decodeBinsEP(3);
  tensor.numDims = numDims;
  ////printf("Decoded number of dimensions: %d\n", numDims);
  binsRead += 3; // 3 bits for numDims

  // decode shape of each dimension
  uint32_t bitlenMinus1 = 0, bitlen = 0;

  for (uint32_t i = 0; i < numDims; i++)
  {
    //bitlenMinus1 = uae_v(5);
    bitlenMinus1 = reader.readBits(5);
    // bitlenMinus1 = m_BinDecoder.decodeBinsEP(5);
    bitlen = bitlenMinus1 + 1;
   // ////printf("Decoded bitlen for dimension %d: %d, -1:%d\n", i, bitlen, bitlenMinus1);
    //shape[i] = uae_v(bitlen);
    shape[i] = reader.readBits(bitlen);
    // shape[i] = m_BinDecoder.decodeBinsEP(bitlen);
    ////printf("Decoded dimension %d size: %d\n", i, shape[i]);
    binsRead += 5 + bitlen; // bits used to decode this dimension
  }

  reader.alignToByte(); // align to byte after reading header

 // printf("Decoded tensor header: id=%d type=%d bitwidth=%d numDims=%d totalHeaderBits=%llu\n",
   //   tensor.tensorId, static_cast<uint32_t>(tensor.tensorType), static_cast<uint32_t>(tensor.tensorBitwidth), numDims, binsRead);

  return binsRead;
}


uint64_t BACDecoder::decodeWeightsChunks(int32_t* pWeights , uint32_t numWeights)
{
  int width = getBitwidthFromEnum(m_tensorBitwidth);
  uint64_t scaledBits = 0;

  const uint32_t chunkSize = 2048 ; // small chunk for low RAM = for 32bits = 8KB
  uint32_t numChunks = (numWeights + chunkSize - 1) >> 11; // 11 because of 2048

  uint8_t*startPtr = m_BinDecoder.getByteStreamPtr();
  for (uint32_t c = 0; c < numChunks; c++)
  {
    m_BinDecoder.startBinDecoder(startPtr);
    //printf("Pointer after BAC init: %p\n", m_BinDecoder.getByteStreamPtr());
    m_CtxModeler.resetNeighborCtx();
    

    //printf("Chunk %d decoding start ptr: %p\n", c, m_BinDecoder.getByteStreamPtr());
    //printf("Chunk %d decoding start\n", c);
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
      ////printf ("Tensor decoded as raw EP bins.\n");
      m_BinDecoder.decodeBinTrm(); // read termination bit for the chunk
      m_BinDecoder.finish();
      continue;
    }
    // read local mean flag and value
    bool useMean = m_BinDecoder.decodeBinEP();
    scaledBits += 1;

    int32_t localMean = 0;
    if (useMean) {
        localMean = iae_v(width);
        scaledBits += width;
    }

    // ---------- read k parameter for Rice-Golomb coding
    uint8_t k = uae_v(2); // read k as 2-bit fixed length for simplicity
    scaledBits += 2;
   // printf("Chunk %d decoding with localMean=%d, k=%d\n", c, localMean, k);


    // --------------- BAC decode weights
    for (uint32_t i = start; i < end; i++)
    {
      int32_t decodedVal = 0;
      scaledBits += decodeWeightVal(decodedVal, k); 

      pWeights[i] =  decodedVal + localMean;
     // ////printf("Decoded weight %d: value=%d\n", i,  pWeights[i]);
      m_CtxModeler.updateNeighborCtx(decodedVal);
      //printf("CHUNK[%d] - Decoded value %d\n", c, pWeights[i]);

    }
    m_BinDecoder.decodeBinTrm(); // read termination bit for the chunk
    //  printf("Chunk %d decoding end ptr before finish: %p\n", c, m_BinDecoder.getByteStreamPtr());

    m_BinDecoder.finish();

    uint32_t bytesRead = m_BinDecoder.getBytesRead();
    startPtr += bytesRead; // move startPtr for next chunk
    //printf("Chunk %d decoding end ptr after finish: %p\n", c, m_BinDecoder.getByteStreamPtr());
  }
  return scaledBits;
}


uint64_t BACDecoder::decodeWeights(int32_t *pWeights, uint32_t numWeights)
{
  uint32_t startBytes = m_BinDecoder.getBytesRead();

  decodeWeightsChunks(pWeights, numWeights); 

  uint32_t endBytes = m_BinDecoder.getBytesRead();

  return endBytes - startBytes; 

}

//int countd = 0;
uint64_t BACDecoder::decodeWeightVal(int32_t &decodedIntVal, uint8_t k )
{ 
  
  uint64_t bitsUsed = 0;

  const int32_t sigctx = m_CtxModeler.getSigCtxId();
  uint32_t sigFlag = m_BinDecoder.decodeBin(m_CtxStore, sigctx, m_tensorType);
  //if (countd < 10)  // Limit the number of printf statements
    //printf("Decoded sigFlag: %d\n", sigFlag);
  
  bitsUsed += 1; // 1 bit for sigFlag


  decodedIntVal = 0;

  if (!sigFlag)
    return bitsUsed;
  

  // sign 
  int32_t signCtx = m_CtxModeler.getSignFlagCtxId();
  uint32_t signFlag = m_BinDecoder.decodeBin(m_CtxStore, signCtx, m_tensorType);
  //if (countd < 10) printf("Decoded signFlag: %d\n", signFlag);
  bitsUsed += 1; // 1 bit for signFlag

  // branch flag
  uint32_t branchFlag = m_BinDecoder.decodeBin(m_CtxStore, 12, m_tensorType); // assuming context 8 is for branch flag
  bitsUsed += 1; // 1 bit for branch flag

  if (branchFlag)
  {
    // if (countd < 10) printf("Decoded branchFlag: %d (large residual case)\n", branchFlag);
    // large residual case, directly decode remAbsLevel without gtx flags
    uint32_t remAbsLevel = 0;
    bitsUsed += decodeAbsRem(remAbsLevel, k);
    decodedIntVal = signFlag ? -int32_t(remAbsLevel + 5) : int32_t(remAbsLevel + 5);
  // if (countd < 10) printf("Decoded weight value: %d (remAbsLevel=%d)\n", decodedIntVal, remAbsLevel);
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

      //if (countd < 10) printf("Decoded grXFlag: %d (numGreaterFlagsDecoded=%d)\n", grXFlag, numGreaterFlagsDecoded);
    } while (grXFlag && numGreaterFlagsDecoded < m_NumGtxFlags);

    //if (grXFlag) { // last grxFlag means decoded value greater than four
     // remAbsLevel ++;
    //}

    decodedIntVal = remAbsLevel + 1; // add 1 to get the original abs value
    decodedIntVal = signFlag ? -decodedIntVal : decodedIntVal;

    //if (countd < 10) printf("Decoded weight value: %d\n", decodedIntVal);
    //countd++;
    return bitsUsed;
  }
}

int32_t BACDecoder::decodeAbsRem(uint32_t& remainder, uint32_t k)
{
  //printf("==> decodeAbsRem , rem %d k %d \n", remainder, k);
  uint32_t binsUsed = 0;
  uint32_t bitwidth = getBitwidthFromEnum(m_tensorBitwidth);
  uint8_t plusBits = 0;
   //printf(" width %d \n", bitwidth);

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
   //printf ("msb1 %d msb2 %d plusb %d \n", msb1, msb2, plusBits);

  uint32_t msb3 = 0, msb4 = 0, msb5 = 0, msb6 = 0;
  if (m_tensorBitwidth == TensorBitwidth::BW_12) {
    msb3 = m_BinDecoder.decodeBin(m_CtxStore, 8, m_tensorType);
    msb4 = m_BinDecoder.decodeBin(m_CtxStore, 9, m_tensorType);
    binsUsed += 2; // 2 bits for MSBs
    plusBits += 2;
    //printf ("msb3 %d msb4 %d plusb %d \n", msb3, msb4, plusBits);
  } else if (m_tensorBitwidth >= TensorBitwidth::BW_16) {
      msb3 = m_BinDecoder.decodeBin(m_CtxStore, 8, m_tensorType);
      msb4 = m_BinDecoder.decodeBin(m_CtxStore, 9, m_tensorType);
      msb5 = m_BinDecoder.decodeBin(m_CtxStore, 10, m_tensorType);
      msb6 = m_BinDecoder.decodeBin(m_CtxStore, 11, m_tensorType);
      binsUsed += 4; // 4 bits for MSBs
      plusBits += 4;
      //printf ("msb3 %d msb4 %d msb5 %d msb6 %d plusb %d \n", msb3, msb4, msb5, msb6, plusBits);
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
  //printf(" q%d kupd%d r%d \n", q, k_upd, r);


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

  //printf("finish: remainder %d lowermsk %d lower %d \n", remainder, lowerMask, lower);
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

void BACDecoder::finishBac()
{
    m_BinDecoder.finish();
;
}
