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
#include "Utils/global_logger.h"


static constexpr uint32_t MAX_TENSORS_BITS = 12;   // allows up to 4096 tensors

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
    m_BinEncoder.startBinEncoder();
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
uint64_t BACEncoder::encodeTensorHeader( const uint32_t* shape, uint32_t numDims, const uint16_t tensorId)
{
    uint64_t binsUsed = 0;
   // if (tensorId > 600){
    //  //printf("Encoding tensor header... Tensor id: %d \n", tensorId);
    //  //printf("Type: %d Width: %d \n", static_cast<uint16_t>(m_tensorType), static_cast<uint16_t>(m_tensorBitwidth));
    //  //printf("Converted width: %d\n", getBitwidthFromEnum(m_tensorBitwidth));
    //}
    
    // encode tensor id
    m_BinEncoder.encodeBinsEP(tensorId, MAX_TENSORS_BITS); // 
    binsUsed += MAX_TENSORS_BITS;
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
      //  //printf("CLZ results: %d\n", __builtin_clz(dimSize));
        m_BinEncoder.encodeBinsEP(bitlen-1, 5);
        //uae_v(5, bitlen - 1);
        m_BinEncoder.encodeBinsEP(dimSize, bitlen);
        //uae_v(bitlen, dimSize);
        shapeBits += 5 + bitlen; // bits used to encode this dimension
      //  //printf("Encoded dimension %d: size=%d, bitlen=%d\n", i, dimSize, bitlen);
        binsUsed += 5 + bitlen;
    }


   // //printf("==> encodeTensorHeader returning with binsUsed=%llu\n", binsUsed);
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
 // //printf("==> iae_v called with v=%d, value=%d\n", v, value);
    uint32_t pattern = uint32_t(value) & (uint32_t(0xFFFFFFFF) >> (32-v));
    ////printf("==> iae_v: pattern=0x%X\n", pattern);
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
/* uint32_t BACEncoder::encodeAbsRem( int32_t value, uint16_t k)
  {
    LOG_LINE(g_logger, "========> EncWeight: xEncRemAbs value=" + std::to_string(value) + ", k=" + std::to_string(k));
    uint32_t scaledBits           = 0;
    uint8_t minusBits = 0;

    uint32_t bitwidth = getBitwidthFromEnum(m_tensorBitwidth);
    //printf(" width %d \n", bitwidth);

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
    LOG_LINE(g_logger, "msb1=" + std::to_string(msb1) + ", msb2=" +  std::to_string(msb2));

    uint32_t msb3 = 0, msb4 = 0, msb5 =0, msb6 =0;

    if (m_tensorBitwidth == TensorBitwidth::BW_12) {
      msb3 = (value >> (bitwidth - 3)) & 0x1;
      scaledBits += m_BinEncoder.encodeBin( msb3, m_CtxStore, 8, m_tensorType );
      msb4 = (value >> (bitwidth - 4)) & 0x1;
      scaledBits += m_BinEncoder.encodeBin( msb4, m_CtxStore, 9, m_tensorType );
      minusBits += 2;
      //printf ("msb3 %d msb4 %d minusb %d \n", msb3, msb4, minusBits);
      LOG_LINE(g_logger, "msb3=" + std::to_string(msb3) + ", msb4=" +  std::to_string(msb4));
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
      //printf ("msb3 %d msb4 %d msb5 %d msb6 %d minusb %d \n", msb3, msb4, msb5, msb6, minusBits);
      LOG_LINE(g_logger, "msb3=" + std::to_string(msb3) + ", msb4" +  std::to_string(msb4) + "msb5=" + std::to_string(msb5) + ", msb6=" +  std::to_string(msb6));
    }

    uint32_t baseMask = (1 << (bitwidth - minusBits)) - 1;
    uint32_t value_no_msb = value & baseMask;

    uint8_t k_upd = k+1;
    uint32_t q = value_no_msb >> k_upd;
    uint32_t r = value_no_msb & ((1 << k_upd) - 1);

    //printf(" basemsk %d valnomsb %d kup %d q %d r %d\n", baseMask, value_no_msb, k_upd, q, r);
    LOG_LINE(g_logger, "basemask=" + std::to_string(baseMask) + ", value no msb=" + std::to_string(value_no_msb) 
              + ", k_upd=" + std::to_string(k_upd) + ", q=" + std::to_string(q) + ", r=" + std::to_string(r));

    // unary prefix
    for(uint32_t i = 0; i < q; i++){
      m_BinEncoder.encodeBinEP(1);
      scaledBits += 1;
    }
    m_BinEncoder.encodeBinEP(0); // terminating bit
    scaledBits += 1;

    // encode suffix
    m_BinEncoder.encodeBinsEP(r, k_upd);
    scaledBits += k_upd;

    //printf("scaled bits %d\n", scaledBits);
    return scaledBits;
  } */

  uint32_t BACEncoder::encodeRice(uint32_t value, uint8_t k){
    uint32_t bits = 0;

    uint32_t q = value >> k;
    uint32_t r = value & ((1u << k) - 1);

    for(uint32_t i=0; i<q; i++){
        m_BinEncoder.encodeBinEP(1);
        bits++;
    }

    m_BinEncoder.encodeBinEP(0);
    bits++;

    m_BinEncoder.encodeBinsEP(r, k);
    bits += k;

    //LOG_LINE(g_logger,         "Rice value=" + std::to_string(value) +        ", k=" + std::to_string(k) +        ", q=" + std::to_string(q) +        ", r=" + std::to_string(r));

    return bits;
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
//      +-- greater than loop
//           |
//           +-- small magnitude part from 1 to 7
//           |     
//           |
//           +-- large magnitude part in groups 15 -> 31 -> 63
//           |
//           |
//           +-- rice coding of the remainder 
//
// Parameters:
//   value   Quantized residual value
//   k       Rice-style suffix parameter
//
// Returns:
//   Number of bins used to encode this weight.
//--------------------------------------------------------------
uint32_t BACEncoder::encodeWeightBAC( int32_t value, uint8_t pred){
   // LOG_LINE(g_logger, "=========> encodeWeightsBAC: value=" + std::to_string(value) + ", k=" + std::to_string(k));

    uint32_t sigFlag        = value != 0 ? 1 : 0;
    int32_t  sigctx         = m_CtxModeler.getSigCtxId( );
    uint32_t scaledBits     = m_BinEncoder.encodeBin(sigFlag, m_CtxStore, sigctx, m_tensorType, pred);
    //printf("sigflag %d \n", sigFlag);
   // LOG_LINE(g_logger, "sigflag= " + std::to_string(sigFlag) + ", sig ctx=" + std::to_string(sigctx));
    
    if (sigFlag){
      uint32_t signFlag = value < 0 ? 1 : 0;

      //signCtx = m_CtxModeler.getSignFlagCtxId();
      //scaledBits += m_BinEncoder.encodeBin(signFlag, m_CtxStore, signCtx, m_tensorType); 
      scaledBits += m_BinEncoder.encodeBinEP(signFlag); // encode sign as EP bin
      
      //printf("signflag %d \n", sigFlag);
     // LOG_LINE(g_logger, "signflag=" + std::to_string(signFlag));    

      uint32_t remAbsLevel = abs(value) - 1;
      //printf("remabs %d \n", remAbsLevel);

      uint32_t riceOffset = 0;
      bool encodeRiceflag = false;

      // greater than (GT) loop from fine to coarse groups: contexts G1, GT2-7, GT15, GT31, GT63
      for(const auto& t : table){
          uint8_t GTflag = remAbsLevel > t.threshold;
         // LOG_LINE(g_logger, "GT" + std::to_string(t.threshold+1) +  "=" + std::to_string(GTflag));
          scaledBits +=
              m_BinEncoder.encodeBin(GTflag, m_CtxStore, t.ctxId, m_tensorType, pred);
          if(!GTflag)
            break;
          
          if(t.riceOffset){
              riceOffset = t.riceOffset;
              encodeRiceflag = true;
          }
          
      }

      if(encodeRiceflag){
        uint8_t k = 0;
        switch (riceOffset){
          case 11:
            k = 1; break;
          case 15:
            k = 2; break;
          case 31:
            k = 3; break;
          case 63:
            k = 4; break;
          default:
            k = 1; break;
        }
        scaledBits += encodeRice(remAbsLevel-riceOffset,k);
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
uint64_t BACEncoder::encodeWeightsChunks( const int32_t* pWeights, uint32_t numWeights){
   // LOG_LINE(g_logger, "=============> encodeWeightsChunks: numWeights=" + std::to_string(numWeights));
    uint64_t scaledBits = 0;
    int width = getBitwidthFromEnum(m_tensorBitwidth);

    //m_estimator.printTable();

    const uint32_t chunkSize = 2048 ; // small chunk for low RAM = for 32bits =~ 65KB 
    uint32_t numChunks = (numWeights + chunkSize - 1) >> 11; // shift for efficiency

    bool skipChunk;

    for (uint32_t c = 0; c < numChunks; c++){
      m_CtxModeler.resetNeighborCtx();

      uint32_t start = c * chunkSize;
      uint32_t end   = std::min(start + chunkSize, numWeights);
      uint32_t len   = end - start;


      // ---- pass 1:compute local mean ----
      int64_t sum = 0;
      for (uint32_t i = start; i < end; i++)
          sum += pWeights[i];

     // uint32_t shift = std::ceil(std::log2(len));
      int32_t localMean = sum / len;

    
      // ---------- pass 2: evaluate predictors -------------
      double bestCost = std::numeric_limits<double>::max();
      Predictor pred= PRED_NONE;

      double predictorCost[3];
      //uint8_t predictorK[3];

      for(int predictor = PRED_NONE; predictor <= PRED_NEIGHBOR; predictor++){
        m_CtxModeler.resetNeighborCtx();
        double cost = 0.0;
        
        for (uint32_t i = start; i < end; i++){
          int32_t r;

          switch (predictor){
          case PRED_NONE:
            r = pWeights[i];
            break;
          case PRED_MEAN:
            r = pWeights[i] - localMean;
            break;
          case PRED_NEIGHBOR:
            if (i == start)
              r = pWeights[i];
            else 
              r = pWeights[i] - pWeights[i-1];
            break;
          }
      
          /// compute bins per element (rough bit estimation)
          cost += estimateWeightBAC(r, predictor);
          predictorCost[predictor] = cost;
          m_CtxModeler.updateNeighborCtx(r);  
        }

        if (cost < bestCost){
          bestCost = cost;
          pred = (Predictor)predictor;
        }
    }

      double bitsPerElement = bestCost / len;
      skipChunk = (bitsPerElement > (width*0.98)); // not sure

      //send skip flag
      m_BinEncoder.encodeBinEP(skipChunk);
      scaledBits += 1;

      /*
      LOG_LINE(
        g_logger,
        "bitwidth=" + std::to_string(width) +
        ", predictor=" + std::to_string(pred) +
        ", costNone=" + std::to_string(predictorCost[PRED_NONE]) +
        ", costMean=" + std::to_string(predictorCost[PRED_MEAN]) +
        ", costNeighbor=" + std::to_string(predictorCost[PRED_NEIGHBOR]) +
        ", bestK=" + std::to_string(bestK) +
        ", bitsPerElement=" + std::to_string(bitsPerElement) +
        ", localMean=" + std::to_string(localMean) +
        ", skip=" + std::to_string(skipChunk)
      );       */

      if (skipChunk){
        ////printf("Skipping BAC chunk encoding. Encoding as raw EP bins instead...\n");
        for (uint32_t c = start; c < end; c++){
          iae_v(width, pWeights[c]);
          //m_BinEncoder.encodeBinsEP(pWeights[c], width);
          scaledBits += width;
        }
        continue;
      }

      // send mean flag and mean value
      //m_BinEncoder.encodeBinEP(useLocalMean ? 1 : 0);
      //scaledBits += 1;
      uae_v(2, pred); // send predictor type
      scaledBits += 2;

      if (pred == PRED_MEAN) {
          iae_v(width, localMean);
          scaledBits += width;
      } 
   

      // ------------ BAC encode ----------------
      m_CtxModeler.resetNeighborCtx();
      for (uint32_t i = start; i < end; i++){
         // LOG_LINE(g_logger, "Value for BAC before mean/neighbor:" + std::to_string(pWeights[i]));
          int32_t value;
          switch (pred){
          case PRED_NONE:
              value = pWeights[i];
              break;
          case PRED_MEAN:
              value = pWeights[i] - localMean;
              break;
          case PRED_NEIGHBOR:
              if (i == start)
                  value = pWeights[i];
              else
                  value = pWeights[i] - pWeights[i - 1];
              break;
          default:
              value = pWeights[i];
              break;
          }
       //   LOG_LINE(g_logger, "Value for BAC AFTER mean/neighbor:" + std::to_string(value));
          scaledBits += encodeWeightBAC(value, pred);
          m_CtxModeler.updateNeighborCtx(value);
      }
    }
  
    return scaledBits;
}


double BACEncoder::estimateWeightBAC(int32_t residual, uint8_t pred){
    double cost = 0.0;
    //-----------------------------
    // SIG
    //-----------------------------
    cost += m_estimator.estimate( m_CtxModeler.getSigCtxId(),  residual != 0, m_tensorType, pred);
    if(residual == 0)
        return cost;
    //-----------------------------
    // SIGN (EP)
    //-----------------------------
    cost += 1.0;
    //-----------------------------
    // Absolute level
    //-----------------------------
    uint32_t remAbsLevel = std::abs(residual) - 1;
    uint32_t riceOffset = 7;
    //-----------------------------
    // GT flags
    //-----------------------------
    bool encodeRice = false;
    uint8_t lastGT = 0;
    for(const auto &t : table){
        uint8_t gtFlag = remAbsLevel > t.threshold;
        cost += m_estimator.estimate(t.ctxId, gtFlag, m_tensorType, pred);
        if(!gtFlag)
            break;
        if(t.riceOffset){
            riceOffset = t.riceOffset;
            encodeRice = true;
            lastGT = t.threshold;
        }
    }
    //-----------------------------
    // Rice
    //-----------------------------
    uint8_t k;
    if(encodeRice){
        switch (lastGT){
        case 11:
          k = 1;  break;
        case 15:
          k = 2;  break;
        case 31:
          k = 3;  break;
        case 63:
          k = 4;  break;
        default:
          k = 1;  break;
        }

        uint32_t riceValue = remAbsLevel - riceOffset;
        uint32_t q = riceValue >> (k);
        // unary prefix
        cost += q;
        // terminating zero
        cost += 1;
        // suffix
        cost += (k);
    }
    return cost;
}


