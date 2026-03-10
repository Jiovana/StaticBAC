#include "Lib/CommonLib/TypeDef.h"
#include "Lib/EncLib/CABACEncoder.h"
#include <iostream>
#include <math.h>
#include "StaticCoder.h"


uint64_t Encoder::encodeLayer(const TensorMeta& tensor, uint16_t tensorId, uint32_t& headerBits)
{
    const std::vector<int32_t>& qindex = tensor.data;
    const uint32_t* shape = tensor.shape.data();
    uint32_t numDims = tensor.numDims;
    const std::string& tensor_name = tensor.name;
    uint32_t numWeights = qindex.size();
    uint64_t bitsUsed = 0;
    bool skipFlag; 

    //printf("==> encodeLayer called with numWeights=%zu, tensor_name=%s\n", qindex.size(), tensor_name.c_str());
    m_CABACEncoder.setBitwidthAndType(tensor.tensorBitwidth, tensor.tensorType);
    // encode tensor header
    skipFlag = m_CABACEncoder.encodeTensorHeader(qindex.data(), numWeights, shape, numDims, tensor_name, tensorId, bitsUsed);
    headerBits = bitsUsed;

    // encode weights
    bitsUsed += m_CABACEncoder.encodeWeights(qindex.data(), numWeights, skipFlag);
    return bitsUsed;
}

const std::vector<uint8_t>&  Encoder::finishEncoding()
{
  m_CABACEncoder.terminateCabacEncoding();
  return m_Bytestream;
}

// bitstream structure
//numTensors
//[tensorHeader][tensorPayload]
//[tensorHeader][tensorPayload]...
const std::vector<uint8_t>& Encoder::encodeModel(const std::vector<TensorMeta>& modelTensors)
{
    uint64_t totalBits = 0;
    uint16_t tensorId = 0;
    uint32_t headerBits = 0;

    //encode number of tensors
    uint32_t numTensors = modelTensors.size();
    m_CABACEncoder.uae_v(10, numTensors); // 10 bits = 1024 tensors limit. 
    totalBits += 10;

    for (const auto& tensor : modelTensors)
    {
      totalBits += this->encodeLayer(tensor, tensorId, headerBits);
      tensorId++;
    }

  //printf("Finished encodig model. Total encoded bits: %lld\n", totalBits);
  return this->finishEncoding();
}


////////////////////////////////////// DECODER /////////////////////////////////////////////////////

void Decoder::setStream( std::vector<uint8_t>& Bytestream )
{
  m_CABACDecoder.startCabacDecoding( Bytestream.data() );
}

void Decoder::decodeLayer(TensorMeta& tensor)
{
    //printf("==> decodeLayer called\n");

    uint32_t shape[8] = {0}; // assuming max 8 dimensions
    uint32_t numDims = 0;
    
    // Decode header
    m_CABACDecoder.decodeTensorHeader(shape,numDims,tensor);
    // Copy shape array into vector
    tensor.shape.assign(shape, shape + numDims);

    // Compute number of weights
    uint32_t numWeights = 1;
    for (uint32_t i = 0; i < numDims; i++)
        numWeights *= shape[i];

    //printf("Decoded tensor header: numDims=%u, numWeights=%u\n", numDims, numWeights);

    // Resize tensor data to hold decoded weights
    tensor.data.resize(numWeights);
    int32_t* pWeights = tensor.data.data();

    // Decode weights
    uint64_t decodedBins = m_CABACDecoder.decodeWeights(pWeights, numWeights);
    //printf("Total decoded bins: %llu\n", decodedBins);
}

uint32_t Decoder::finishDecoding()
{
  uint32_t bytesRead = m_CABACDecoder.terminateCabacDecoding();
  return bytesRead;
}

/// @brief  this function is not finished yet!
/// @param modelTensors 
void Decoder::decodeModel(std::vector<TensorMeta>& modelTensors)
{
    // Decode number of tensors first
    uint32_t numTensors = m_CABACDecoder.uae_v(10); // up to 1024 tensors

    //printf("Decoding model with %u tensors\n", numTensors);

    modelTensors.resize(numTensors);

    for (uint32_t i = 0; i < numTensors; i++)
    {
        //printf("Decoding tensor %u\n", i);

        decodeLayer(modelTensors[i]);   // fills TensorMeta directly
    }

    this->finishDecoding();
}

