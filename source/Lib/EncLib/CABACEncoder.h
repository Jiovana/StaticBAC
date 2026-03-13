#ifndef BAC_ENC_H
#define BAC_ENC_H

#include "../CommonLib/ContextModel.h"
#include "../CommonLib/ContextModeler.h"
#include "BinEncoder_simple.h"
#include <bitset>
#include <limits>
#include <iostream>
#include <algorithm>
//#include "Utils/global_logger.h" 
#include <sstream>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cstdint>
#include <string>

////////////////////////////////////////////////////////////
/// CABACEncoder
///
/// High-level encoder responsible for encoding tensor
/// weights using binary arithmetic coding.
///
/// Workflow:
///   startBacEncoding()
///   encodeTensorHeader()
///   encodeWeights()
///   terminateBacEncoding()
////////////////////////////////////////////////////////////
class BACEncoder
{
  public:
    BACEncoder() = default;
    ~BACEncoder() = default;

    ////////////////////////////////////////////////////////////
    /// Encoder control
    ////////////////////////////////////////////////////////////

    /// Initialize BAC encoder and attach output bytestream
    void startBacEncoding(std::vector<uint8_t>* pBytestream);

    /// Finish encoding and flush remaining bits
    void terminateBacEncoding();

    /// Initialize context models
    void initCtxMdls(uint32_t numGtxFlags);

    ////////////////////////////////////////////////////////////
    /// Bitstream helpers
    ////////////////////////////////////////////////////////////

    /// Encode signed integer using fixed bitwidth
    void iae_v(uint8_t v, int32_t value);

    /// Encode unsigned integer using fixed bitwidth
    void uae_v(uint8_t v, uint32_t value);

    ////////////////////////////////////////////////////////////
    /// Tensor encoding
    ////////////////////////////////////////////////////////////

    /// Encode tensor metadata header
    uint64_t encodeTensorHeader(
        const int32_t* pWeights,
        uint32_t numWeights,
        const uint32_t* shape,
        uint32_t numDims,
        const std::string& tensor_name,
        uint16_t tensorId);

    /// Encode tensor weights
    uint64_t encodeWeights(const int32_t* pWeights, uint32_t numWeights);

    /// Configure tensor parameters
    inline void setBitwidthAndType(TensorBitwidth bitwidth, TensorType type) {
      m_tensorBitwidth = bitwidth;
      m_tensorType = type;
    }

  private:
    ////////////////////////////////////////////////////////////
    /// Core BAC weight coding primitives
    ////////////////////////////////////////////////////////////

    uint32_t encodeWeightBAC(int32_t value, uint8_t k);
    uint32_t encodeAbsRem(int32_t value, uint16_t k);

    ////////////////////////////////////////////////////////////
    /// Main weight encoding algorithm
    ////////////////////////////////////////////////////////////

    uint64_t encodeWeightsChunks(const int32_t* pWeights, uint32_t numWeights);

  private:

    ////////////////////////////////////////////////////////////
    /// BAC state
    ////////////////////////////////////////////////////////////

    StaticCtx      m_CtxStore;
    ContextModeler m_CtxModeler;
    BinEnc         m_BinEncoder;

    uint32_t       m_NumGtxFlags;

    TensorBitwidth m_tensorBitwidth;
    TensorType     m_tensorType;

    int32_t        m_TensorMean;
    bool           m_useMean;
};

#endif
