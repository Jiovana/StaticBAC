//==============================================================
// BACEncoder.cpp
//
// Implementation of a not adaptive binary arithmetic
// encoder for neural network tensor compression.
//
// This encoder uses a simplified CABAC-style structure with:
//
//  - significance flags
//  - sign flags
//  - greater-than-X unary coding
//  - Rice-style remainder coding
//
// Weights are processed in chunks to enable adaptive coding
// decisions and optional bypass when compression is
// predicted to be ineffective.
//==============================================================
#include "CABACEncoder.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <sstream>
//#include "Utils/global_logger.h"

//--------------------------------------------------------------
// startBacEncoding
//
// Initializes the BAC encoder and attaches the output byte
// stream buffer.
//
// This resets the arithmetic encoder state and prepares the
// encoder to begin writing bins.
//
// Parameters:
//   pBytestream  Pointer to the output byte buffer where the
//                encoded bitstream will be written.
//
// Notes:
//   - Must be called before any encoding operations.
//   - Resets tensor mean to zero for safety.
//--------------------------------------------------------------
void BACEncoder::startBacEncoding( std::vector<uint8_t>* pBytestream)
{
    m_BinEncoder.setByteStreamBuf(pBytestream);
    m_TensorMean = 0;
    m_BinEncoder.startBinEncoder();
}

//--------------------------------------------------------------
// initCtxMdls
//
// Initializes the context modeler used by the BAC encoder.
//
// Does not do much since contexts are static, just sets neighbor
// to zero and the GtxFlags, which are constant anyway.
//
// Parameters:
//   numGtxFlags  Maximum number of "greater-than-X" flags
//                used in the unary coding of small magnitudes.
//
// Notes:
//   Should be called once before encoding weights.
//--------------------------------------------------------------
void BACEncoder::initCtxMdls(uint32_t numGtxFlags)
{
  m_NumGtxFlags = numGtxFlags;
  m_CtxModeler.init(numGtxFlags);
}

//--------------------------------------------------------------
// encodeTensorHeader
//
// Encodes metadata describing a tensor before its weights
// are encoded. This header allows the decoder to reconstruct
// the tensor structure and decoding parameters.
//
// Encoded information:
//   - Tensor ID
//   - Tensor type (weights or bias)
//   - Quantization bitwidth
//   - Number of tensor dimensions
//   - Shape of each dimension
//   - Optional tensor mean (for residual coding)
//
// Parameters:
//   pWeights     Pointer to tensor weight values
//   numWeights   Total number of weights
//   shape        Array describing tensor dimensions
//   numDims      Number of tensor dimensions
//   tensor_name  Debug/logging name of tensor
//   tensorId     Unique tensor identifier
//
// Returns:
//   Number of bins used to encode the header.
//
// Notes:
//   The mean is estimated using an approximate power-of-two
//   division and transmitted only if its magnitude exceeds a
//   small threshold.
//--------------------------------------------------------------
uint64_t BACEncoder::encodeTensorHeader(const int32_t* pWeights, uint32_t numWeights, const uint32_t* shape, uint32_t numDims, const std::string& tensor_name, const uint16_t tensorId)
{
    uint64_t binsUsed = 0;
    // encode tensor id
    m_BinEncoder.encodeBinsEP(tensorId, 10); // 10 bits = 1024 tensors
    binsUsed +=10;
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
      //  printf("CLZ results: %d\n", __builtin_clz(dimSize));
        m_BinEncoder.encodeBinsEP(bitlen-1, 5);
        //uae_v(5, bitlen - 1);
        m_BinEncoder.encodeBinsEP(dimSize, bitlen);
        //uae_v(bitlen, dimSize);
        shapeBits += 5 + bitlen; // bits used to encode this dimension
      //  printf("Encoded dimension %d: size=%d, bitlen=%d\n", i, dimSize, bitlen);
        binsUsed += 5 + bitlen;
    }

    int64_t sum = 0;
    int32_t max_abs = 0;
    uint32_t count = 0;

    const uint32_t chunkSize = 2048 ; // small chunk for low RAM ?
    uint32_t numChunks = (numWeights + chunkSize - 1) >> 11;
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
     // printf("Encoding mean: %d with bitwidth: %d\n", mean, bitwidth);
      iae_v(bitwidth, mean); 
      binsUsed += bitwidth; // account for bits used to encode mean
    //  printf("Encoded mean: %d\n", mean);
    } else {
      m_TensorMean = 0; // if not using mean, set it to zero for encoding residuals
    //    printf("Mean not used, mean value: %d\n", mean);
    }

   // printf("==> encodeTensorHeader returning with binsUsed=%llu\n", binsUsed);
    return binsUsed;
}

//--------------------------------------------------------------
// iae_v  (signed integer arithmetic encoding)
//
// Encodes a signed integer value using a fixed number of
// equiprobable (EP) bins.
//
// Parameters:
//   v      Number of bits used to represent the value
//   value  Signed integer value to encode
//
// Notes:
//   The value is truncated to the lowest v bits before
//   encoding. This is typically used for raw residual or
//   fallback coding.
//--------------------------------------------------------------
void BACEncoder::iae_v( uint8_t v, int32_t value )
{
  //PROFILE_SCOPE("iae_v", 0);
 // printf("==> iae_v called with v=%d, value=%d\n", v, value);
    uint32_t pattern = uint32_t(value) & (uint32_t(0xFFFFFFFF) >> (32-v));
  //  printf("==> iae_v: pattern=0x%X\n", pattern);
    m_BinEncoder.encodeBinsEP( pattern, v );
}

//--------------------------------------------------------------
// uae_v  (unsigned integer arithmetic encoding)
//
// Encodes an unsigned integer value using v equiprobable bins.
//
// Parameters:
//   v      Number of bits to encode
//   value  Unsigned integer value
//
// Notes:
//   This is a convenience wrapper around encodeBinsEP()
//   used for fixed-length parameter transmission.
//--------------------------------------------------------------
void BACEncoder::uae_v( uint8_t v, uint32_t value )
{
  //PROFILE_SCOPE("uae_v", 0);
    m_BinEncoder.encodeBinsEP( value, v );
}

//--------------------------------------------------------------
// terminateBacEncoding
//
// Finalizes the arithmetic encoding process and flushes the
// remaining state of the BAC encoder into the output
// bytestream.
//
// Notes:
//   Must be called after all bins have been encoded.
//--------------------------------------------------------------
void BACEncoder::terminateBacEncoding()
{
    //PROFILE_SCOPE("terminateCabacEncoding", 0);
    m_BinEncoder.encodeBinTrm(1);
    m_BinEncoder.finish();
}

//--------------------------------------------------------------
// encodeWeights
//
// Entry point for encoding tensor weights using BAC.
//
// This function delegates the actual encoding work to the
// chunk-based encoder, which processes the tensor in blocks
// to reduce memory usage and allow adaptive coding decisions.
//
// Parameters:
//   pWeights     Pointer to tensor weight values
//   numWeights   Number of weights in the tensor
//
// Returns:
//   Total number of bins produced during encoding.
//--------------------------------------------------------------
uint64_t BACEncoder::encodeWeights(const int32_t *pWeights, uint32_t numWeights)
{
  return encodeWeightsChunks(pWeights, numWeights);
}


//--------------------------------------------------------------
// encodeAbsRem
//
// Encodes the remaining absolute value of a weight after the
// small-magnitude branch has been exceeded.
//
// Coding structure:
//   - Most significant bits encoded with context models
//   - Remaining magnitude encoded using unary prefix and
//     truncated binary suffix
//
// Parameters:
//   value   Remaining absolute value to encode
//   k       Rice-style suffix parameter controlling suffix size
//
// Returns:
//   Number of bins used for encoding.
//
// Notes:
//   This function handles the "large residual" branch of the
//   weight coding scheme.
//--------------------------------------------------------------
uint32_t BACEncoder::encodeAbsRem( int32_t value, uint16_t k)
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
    scaledBits += m_BinEncoder.encodeBin( msb1, m_CtxStore, 6, m_tensorType );
    scaledBits += m_BinEncoder.encodeBin( msb2, m_CtxStore, 7, m_tensorType );
    minusBits += 2;

    uint32_t msb3 = 0, msb4 = 0, msb5 =0, msb6 =0;

    if (m_tensorBitwidth == TensorBitwidth::BW_12) {
      msb3 = (value >> (bitwidth - 3)) & 0x1;
      scaledBits += m_BinEncoder.encodeBin( msb3, m_CtxStore, 8, m_tensorType );
      msb4 = (value >> (bitwidth - 4)) & 0x1;
      scaledBits += m_BinEncoder.encodeBin( msb4, m_CtxStore, 9, m_tensorType );
      minusBits += 2;
    } else if (m_tensorBitwidth >= TensorBitwidth::BW_16) {
      msb3 = (value >> (bitwidth - 3)) & 0x1;
      scaledBits += m_BinEncoder.encodeBin( msb3, m_CtxStore, 8, m_tensorType );
      msb4 = (value >> (bitwidth - 4)) & 0x1;
      scaledBits += m_BinEncoder.encodeBin( msb4, m_CtxStore, 9, m_tensorType );
      msb5 = (value >> (bitwidth - 5)) & 0x1;
      scaledBits += m_BinEncoder.encodeBin( msb5, m_CtxStore, 10, m_tensorType );
      msb6 = (value >> (bitwidth - 6)) & 0x1;
      scaledBits += m_BinEncoder.encodeBin( msb6, m_CtxStore, 11, m_tensorType );
      minusBits += 4;
    }

    uint32_t baseMask = (1 << (bitwidth - minusBits)) - 1;
    uint32_t value_no_msb = value & baseMask;

    uint8_t k_upd = k+1;
    uint32_t q = value_no_msb >> k_upd;
    uint32_t r = value_no_msb & ((1 << k_upd) - 1);

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
 
    return scaledBits;
  }


//--------------------------------------------------------------
// encodeWeightBAC
//
// Encodes a single quantized weight value using the BAC
//  coding scheme.
//
// Coding structure:
//
//   sigFlag
//      |
//      +-- signFlag
//      |
//      +-- branchFlag
//           |
//           +-- small magnitude branch (<= 5)
//           |     -> sequence of greater-than-X flags
//           |
//           +-- large magnitude branch (> 5)
//                 -> encodeAbsRem()
//
// Parameters:
//   value   Quantized residual value
//   k       Rice-style suffix parameter
//
// Returns:
//   Number of bins used to encode this weight.
//
// Notes:
//   
//--------------------------------------------------------------
uint32_t BACEncoder::encodeWeightBAC( int32_t value, uint8_t k)
  {
    uint32_t sigFlag        = value != 0 ? 1 : 0;
    int32_t  sigctx         = m_CtxModeler.getSigCtxId( );
    uint32_t scaledBits     = m_BinEncoder.encodeBin(sigFlag, m_CtxStore, sigctx, m_tensorType);
    
    if (sigFlag)
    {
      uint32_t signFlag = value < 0 ? 1 : 0;
      int32_t signCtx;

      signCtx = m_CtxModeler.getSignFlagCtxId();
      scaledBits += m_BinEncoder.encodeBin(signFlag, m_CtxStore, signCtx, m_tensorType);     

      uint32_t remAbsLevel = abs(value) - 1;

      if (abs(value) > 5){
        // bypass gtx flags and directly encode remAbsLevel using xEncRemAbs
        scaledBits += m_BinEncoder.encodeBin(1, m_CtxStore, 12, m_tensorType); // set branch flag to 1 to indicate large residual
        remAbsLevel -= 5; // we can subtract 5 here because values <=5 are handled in the small branch, this way we encode a smaller number in xEncRemAbs which is more efficient
        scaledBits += encodeAbsRem( remAbsLevel, k); 
      } else {
        scaledBits += m_BinEncoder.encodeBin(0, m_CtxStore, 12, m_tensorType); // set branch flag to 0 to indicate small residual

        uint32_t grXFlag = remAbsLevel ? 1 : 0; //greater1
        int32_t ctxIdx;

        ctxIdx = m_CtxModeler.getGtxCtxId( signFlag);
        scaledBits += m_BinEncoder.encodeBin(grXFlag, m_CtxStore, ctxIdx, m_tensorType);

        uint32_t numGreaterFlagsCoded = 1;
        //printf("==> EncWeight: signctx=%d, ctxidx=%d, signFlag=%d, grXFlag=%d, scaledBits=%d\n", signCtx, ctxIdx, signFlag, grXFlag, scaledBits);
        while (grXFlag && (numGreaterFlagsCoded < m_NumGtxFlags) )
        {
          remAbsLevel--;
          grXFlag = remAbsLevel ? 1 : 0;
          ctxIdx =  m_CtxModeler.getGtxCtxId(signFlag);         
          scaledBits += m_BinEncoder.encodeBin(grXFlag, m_CtxStore, ctxIdx, m_tensorType);        
          numGreaterFlagsCoded++;
          //printf("==> EncWeight: numGreaterFlagsCoded=%d, ctxidx=%d, remAbsLevel=%d, grXFlag=%d, scaledBits=%d\n", numGreaterFlagsCoded, ctxIdx, remAbsLevel, grXFlag, scaledBits);
        }

      }
      
      
    }
    return scaledBits;
  }

//--------------------------------------------------------------
// encodeWeightsChunks
//
// Encodes tensor weights in fixed-size chunks to limit memory
// usage and allow per-chunk coding decisions.
//
// Processing steps for each chunk:
//
//   1. Compute residual statistics
//   2. Estimate coding efficiency
//   3. Decide whether to skip BAC coding
//   4. If skipped:
//        encode weights using raw EP bins
//      Else:
//        perform BAC coding
//
// Parameters:
//   pWeights     Pointer to tensor weight values
//   numWeights   Total number of weights
//
// Returns:
//   Total number of bins produced.
//
// Notes:
//   Chunk processing enables adaptive parameter selection
//   (e.g., Rice parameter k) and optional bypass of BAC
//   coding when compression is predicted to be ineffective.
//--------------------------------------------------------------
uint64_t BACEncoder::encodeWeightsChunks( const int32_t* pWeights, uint32_t numWeights)
{
    uint64_t scaledBits = 0;
    int32_t localMean = m_TensorMean;
    int width = getBitwidthFromEnum(m_tensorBitwidth);

    const uint32_t chunkSize = 2048 ; // small chunk for low RAM = for 32bits =~ 65KB 
    uint32_t numChunks = (numWeights + chunkSize - 1) >> 11; // shift for efficiency

    double avgBPE = 0.0;

    bool skipChunk;

    std::vector<int32_t> scaledBuf(chunkSize);
    for (uint32_t c = 0; c < numChunks; c++)
    {
      m_CtxModeler.resetNeighborCtx();

      uint32_t start = c * chunkSize;
      uint32_t end   = std::min(start + chunkSize, numWeights);
      uint32_t len   = end - start;

      int64_t sumRes = 0;
      int32_t residual = 0;

      // ---------- pass 1: residual + meanResidual -------------
      for (uint32_t i = start; i < end; i++)
      {
          residual = pWeights[i] - localMean;
          sumRes += std::abs(residual);
      }

      // --------------- mean abs residual ---------------
      int32_t meanRes = sumRes / len;
      if (meanRes == 0) meanRes = 1;

      // ---------- compute K --------------
      uint8_t k = 0;
      if      (meanRes < 8)       k = 0;
      else if (meanRes < 32)      k = 1;
      else if (meanRes < 256)     k = 2;
      else if (meanRes < 1024)    k = 3;
      else                        k = 3;

      uint8_t shift = getShiftFromMeanAndK(m_tensorBitwidth, m_TensorMean, k);

      // ------------ pass 2 - histogram on scaled residuals 
      uint64_t estBits = 0;
      
      for (uint32_t i = start; i < end; i++)
      {
          int32_t residual = pWeights[i] - localMean;

          int32_t scaled;
          if (shift > 0)
              scaled = (residual + (residual >= 0 ? (1 << (shift-1)) : (1 << (shift-1)))) >> shift;
          else
              scaled = residual;

          scaledBuf[i - start] = scaled;

          /// compute bins per element (rough bit estimation)
          uint32_t absScaled = std::abs(scaled);
          if (absScaled == 0) estBits += 1 ; // sig only (minimal)
          else if (absScaled <= 5) {
              estBits += 1 + 1 + absScaled; // sig + sign + branch + grXFlags (branch included in absscaled)
          } else {
              // rough estimate: MSBs + 1 unary + k + suffix
              // here we can use xEncRemAbs logic without actual bin encoder calls
              uint32_t minusBits = 2; // first 2 MSBs
              if (width == 12) minusBits += 2;
              else if (width >= 16) minusBits += 4;
              uint32_t remAbs = absScaled - 5;
              uint32_t q = remAbs >> (k+1);
              //uint32_t r = remAbs & ((1 << (k+1)) - 1);
              estBits += minusBits + 1 + q + 1 + (k+1); // MSBs + branch + unary + 0 term + suffix
          }
      }
      estBits = round(estBits * 0.9); // reduce 10% to account for bac coded bins (pessimist)

      double binsPerElement = double(estBits) / len;

      avgBPE += binsPerElement * len;

      double normBPE = binsPerElement / width;

      skipChunk = (normBPE > 0.98); // not sure
      //bool skipChunk = (inneficiency > 1.03);

      //send skip flag
      m_BinEncoder.encodeBinEP(skipChunk ? 1 : 0);
      scaledBits += 1;

      if (skipChunk){
        //printf("Skipping BAC chunk encoding. Encoding as raw EP bins instead...\n");
        for (uint32_t c = start; c < end; c++){
          iae_v(width, pWeights[c]);
          //m_BinEncoder.encodeBinsEP(pWeights[c], width);
          scaledBits += width;
        }
        continue;
      }
    
      // send k  
      uae_v(2, k); // send k as 2-bit 
      scaledBits += 2; // account for bits used to encode k 

      // ------------ pass 3: bac encoding ----------------
      for (uint32_t i = start; i < end; i++)
      {
        int32_t scaled = scaledBuf[i-start];

        scaledBits += encodeWeightBAC(scaled, k);
        m_CtxModeler.updateNeighborCtx(scaled);  

        }

    }
    // ---------- print tensor-level averages ----------
    avgBPE     /= numWeights;
   // std::cout << "Tensor-level avgEntropy=" << avgEntropy
   //           << ", avgBPE=" << avgBPE << ", Skip? " << skipChunk << "\n";

    return scaledBits;
}





