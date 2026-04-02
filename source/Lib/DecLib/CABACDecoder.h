#ifndef __BACDEC__
#define __BACDEC__

#include <vector>
#include <algorithm>

#include "CommonLib/ContextModel.h"
#include "CommonLib/ContextModeler.h"
#include "CommonLib/Quant.h"
#include "CommonLib/Scan.h"
#include "BinDecoder.h"
#include "../CommonLib/TypeDef.h"

class BACDecoder
{
public:
    BACDecoder() = default;
    ~BACDecoder() = default;

    /* Initializes CABAC decoder with input bytestream */
    void     startBacDecoding    ( uint8_t* pBytestream );
    /* Initializes context models */
    void     initCtxModels           ( uint32_t cabac_unary_length );
    /* Finalizes decoding */
    uint32_t terminateBacDecoding();

    void finishBac();

    uint8_t* getBytestreamPtr();
    void setByteStreamPtr(uint8_t* ptr);
    void setBytesRead(uint32_t bytesRead);

    /* Signed EP bin decoding */
    int32_t  iae_v                 ( uint8_t v );
    /* Unsigned EP bin decoding */
    uint32_t uae_v                 ( uint8_t v );
    /* Decodes tensor header and metadata */
    uint64_t    decodeTensorHeader     ( uint32_t* shape, uint32_t& numDims, TensorMeta &tensor, BitstreamReader& reader );
    /* Decodes tensor weights */
    uint8_t*    decodeWeights          ( int32_t* pWeights, uint32_t numWeights, uint8_t* payloadPtr );
    /* Bytes consumed from bitstream */
    uint32_t  getBytesRead();

protected:

  
  uint8_t* decodeWeightsChunks(int32_t* pWeights, uint32_t numWeights, uint8_t* payloadPtr);
  uint64_t decodeWeightVal    ( int32_t &decodedIntVal, uint8_t k );
  int32_t  decodeAbsRem       ( uint32_t& remainder, uint32_t k );
   
private:
    StaticCtx             m_CtxStore;
    ContextModeler        m_CtxModeler;
    BinDec                m_BinDecoder;
    uint32_t              m_NumGtxFlags;
    TensorBitwidth        m_tensorBitwidth;
    TensorType            m_tensorType;
};
#endif // __BACDEC__
