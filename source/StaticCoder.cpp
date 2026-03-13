#include "Lib/CommonLib/TypeDef.h"
#include "Lib/EncLib/CABACEncoder.h"
#include <iostream>
#include <math.h>
#include "StaticCoder.h"

static constexpr uint32_t MAX_TENSORS_BITS = 10;   // allows up to 1024 tensors
static constexpr uint32_t MAX_TENSOR_DIMS  = 8;    // max tensor rank supported

///////////////////////////////////////////////////////////////
///
/// Encode a single tensor layer
///
/// Encodes:
/// 1. Tensor header (metadata)
/// 2. Quantized tensor weights
///
/// @param tensor       Tensor metadata + quantized weights
/// @param tensorId     Sequential tensor identifier
/// @param headerBits   Output number of bits used by header
///
/// @return total number of bits used for this tensor
///
///////////////////////////////////////////////////////////////
uint64_t Encoder::encodeLayer(const TensorMeta& tensor, uint16_t tensorId, uint32_t& headerBits)
{
    const uint32_t numWeights = tensor.data.size();
    uint64_t bitsUsed = 0;

    m_BACEncoder.setBitwidthAndType(tensor.tensorBitwidth, tensor.tensorType);

    // Encode tensor header
    uint64_t headerBitsLocal =
        m_BACEncoder.encodeTensorHeader(
            tensor.data.data(),
            numWeights,
            tensor.shape.data(),
            tensor.numDims,
            tensor.name,
            tensorId);

    headerBits = headerBitsLocal;
    bitsUsed += headerBitsLocal;

    // encode weights
    bitsUsed += m_BACEncoder.encodeWeights(tensor.data.data(), numWeights);
    return bitsUsed;
}

///////////////////////////////////////////////////////////////
///
/// Finalize CABAC encoding and return compressed bytestream
///
/// @return reference to encoded bytestream
///
///////////////////////////////////////////////////////////////
const std::vector<uint8_t>&  Encoder::finishEncoding()
{
  m_BACEncoder.terminateBacEncoding();
  return m_Bytestream;
}


///////////////////////////////////////////////////////////////
///
/// Encode an entire neural network model
///
/// Bitstream structure:
///
/// [numTensors]
/// [tensorHeader][tensorPayload]
/// [tensorHeader][tensorPayload]
/// ...
///
/// @param modelTensors vector containing all tensors to encode
///
/// @return reference to compressed model bytestream
///
///////////////////////////////////////////////////////////////
const std::vector<uint8_t>& Encoder::encodeModel(const std::vector<TensorMeta>& modelTensors)
{
    //encode number of tensors
    const uint32_t numTensors = modelTensors.size();
    m_BACEncoder.uae_v(MAX_TENSORS_BITS, numTensors); // 10 bits = 1024 tensors limit. 

    uint32_t headerBits = 0;
    for (uint16_t tensorId = 0; tensorId < numTensors; tensorId++)
    {
      encodeLayer(modelTensors[tensorId], tensorId, headerBits);
    }

  return this->finishEncoding();
}


///////////////////////////////////////////////////////////////
///
/// Set CABAC decoder input stream
///
/// @param Bytestream compressed bitstream buffer
///
///////////////////////////////////////////////////////////////
void Decoder::setStream( std::vector<uint8_t>& Bytestream )
{
  m_BACDecoder.startBacDecoding( Bytestream.data() );
}

///////////////////////////////////////////////////////////////
///
/// Decode a single tensor layer
///
/// Performs:
/// 1. Tensor header decoding
/// 2. Tensor shape reconstruction
/// 3. Weight decoding
///
/// @param tensor TensorMeta structure to fill
///
///////////////////////////////////////////////////////////////
void Decoder::decodeLayer(TensorMeta& tensor)
{

    uint32_t shape[MAX_TENSOR_DIMS] = {0}; // assuming max 8 dimensions
    uint32_t numDims = 0;
    
    // Decode header
    m_BACDecoder.decodeTensorHeader(shape, numDims, tensor);
    // Copy shape array into vector
    tensor.shape.assign(shape, shape + numDims);

    // Compute number of weights
    uint32_t numWeights = 1;
    for (uint32_t i = 0; i < numDims; i++)
        numWeights *= shape[i];

    // Resize tensor data to hold decoded weights
    tensor.data.resize(numWeights);

    // Decode weights
    m_BACDecoder.decodeWeights(tensor.data.data(), numWeights);

}

///////////////////////////////////////////////////////////////
///
/// Finish CABAC decoding
///
/// @return number of bytes consumed from bitstream
///
///////////////////////////////////////////////////////////////
uint32_t Decoder::finishDecoding()
{
  return m_BACDecoder.terminateBacDecoding();
}

///////////////////////////////////////////////////////////////
///
/// Decode an entire compressed model
///
/// Bitstream structure:
///
/// [numTensors]
/// [tensorHeader][tensorPayload]
/// [tensorHeader][tensorPayload]
/// ...
///
/// @param modelTensors output vector of decoded tensors
///
///////////////////////////////////////////////////////////////
void Decoder::decodeModel(std::vector<TensorMeta>& modelTensors)
{
    // Decode number of tensors 
    uint32_t numTensors = m_BACDecoder.uae_v(MAX_TENSORS_BITS); // up to 1024 tensors

    modelTensors.resize(numTensors);

    for (uint32_t i = 0; i < numTensors; i++)
    {
        decodeLayer(modelTensors[i]);   // fills TensorMeta directly
    }
    finishDecoding();
}

