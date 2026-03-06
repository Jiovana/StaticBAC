#include "Lib/CommonLib/TypeDef.h"
#include "Lib/EncLib/CABACEncoder.h"
#include <iostream>
#include <math.h>
#include "StaticCoder.h"


uint64_t Encoder::encodeLayer(const TensorMeta& tensor)
{
    const std::vector<int32_t>& qindex = tensor.data;
    const uint32_t* shape = tensor.shape.data();
    uint32_t numDims = tensor.numDims;
    const std::string& tensor_name = tensor.name;
    uint32_t numWeights = qindex.size();
    uint64_t bitsUsed = 0;

    printf("==> encodeLayer called with numWeights=%zu, tensor_name=%s\n", qindex.size(), tensor_name.c_str());


    m_CABACEncoder.setBitwidthAndType(tensor.tensorBitwidth, tensor.TensorType);

    // encode tensor header
    bitsUsed += m_CABACEncoder.encodeTensorHeader(qindex.data(), numWeights, shape, numDims, tensor_name);

    // encode weights
    bitsUsed += m_CABACEncoder.encodeWeights(qindex.data(), numWeights);

    return bitsUsed;
}

const std::vector<uint8_t>&  Encoder::finishEncoding()
{
  m_CABACEncoder.terminateCabacEncoding();
  return m_Bytestream;
}

const std::vector<uint8_t>& Encoder::encodeModel(const std::vector<TensorMeta>& modelTensors)
{
    uint64_t totalBits = 0;

    //encode number of tensors
    uint32_t numTensors = modelTensors.size();
    m_CABACEncoder.uae_v(10, numTensors); // 10 bits = 1024 tensors limit. 
    totalBits += 10;

    for (const auto& tensor : modelTensors)
    {
      totalBits += this->encodeLayer(tensor);
    }

  printf("Finished encodig model. Total encoded bits: %lld\n", totalBits);
  return this->finishEncoding();
}






//void encodeModel(const std::vector<TensorMeta>& modelTensors);
//numTensors
//[tensorHeader][tensorPayload]
//[tensorHeader][tensorPayload]


///////////////////////////////////////////////////////////////////////////////////////////

void Decoder::setStream( std::vector<uint8_t>& Bytestream )
{
  m_CABACDecoder.startCabacDecoding( Bytestream.data() );
}

void Decoder::decodeLayer( std::vector<int32_t>& Weights )     
{
  printf("==> decodeLayer called \n");
  int32_t* pWeights   = Weights.data();

  uint32_t shape[8] = {0}; // assuming max 8 dimensions
  uint32_t numDims = 0;

  m_CABACDecoder.decodeTensorHeader(shape, numDims); // we can ignore shape and numDims for now since we have numWeights, but this can be extended to fill shape and numDims if needed

  uint32_t numWeights = 1;
  for (uint32_t i = 0; i < numDims; i++)
      numWeights *= shape[i];

  printf("Decoded tensor header: numDims=%d, numWeights=%d\n", numDims, numWeights);
  
  Weights.resize(numWeights); // ensure Weights vector is sized to hold all decoded weights
  pWeights = Weights.data(); // update pWeights in case resize caused reallocation

  uint64_t decodedBins = 0;
  decodedBins = m_CABACDecoder.decodeWeights(pWeights, numWeights );
  printf("Total decoded  bins: %d\n", decodedBins);

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
    uint32_t numTensors = m_CABACDecoder.uae_v(8);
    modelTensors.resize(numTensors);

    for (uint32_t i = 0; i < numTensors; i++)
    {
        std::vector<int32_t> decodedData;
        decodeLayer(decodedData); // reads tensor header + weights

        modelTensors[i].data = decodedData;
        // Optionally fill shape/numDims if decodeLayer fills it
    }

    this->finishDecoding();
}


