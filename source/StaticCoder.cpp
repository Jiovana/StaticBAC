#include "Lib/CommonLib/TypeDef.h"
#include "Lib/EncLib/CABACEncoder.h"
#include <iostream>
#include <math.h>
#include "StaticCoder.h"

static constexpr uint32_t MAX_TENSORS_BITS = 12;   // allows up to 4096 tensors
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

    //std::cout << "Encoding tensor: " << tensor.name << " with numWeights: " << numWeights << "\n";

    // Encode tensor header
    uint64_t headerBitsLocal =
        m_BACEncoder.encodeTensorHeader(
            tensor.shape.data(),
            tensor.numDims,
            tensorId, m_Bytestream);

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
  //m_BACEncoder.terminateBacEncoding();
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
    // no BAC encoding
    BitstreamWriter writer(&m_Bytestream);
    writer.writeBits(numTensors, MAX_TENSORS_BITS); // 12 bits = 4096 tensors limit
    writer.flushToByte(); // flush after writing numTensors
   // printf("Raw bytes after writing numTensors: ");
   // for (int i = 0; i < 2; i++) printf("%02X ", m_Bytestream[i]);
   // printf("\n");

    printf("Encoding model with %d tensors\n", numTensors);

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
  printf("Bytestream pointer set in decoder set stream: %p\n", Bytestream.data());
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
/// @param ptr pointer to the current position in the bytestream
///
///////////////////////////////////////////////////////////////
uint8_t* Decoder::decodeLayer(TensorMeta& tensor, uint8_t* ptr)
{
    BitstreamReader reader(ptr);

    uint32_t shape[MAX_TENSOR_DIMS] = {0}; // assuming max 8 dimensions
    uint32_t numDims = 0;

    // Decode header
    m_BACDecoder.decodeTensorHeader(shape, numDims, tensor, reader);
    // Copy shape array into vector
    tensor.shape.assign(shape, shape + numDims);

    // --- Move BAC pointer past the raw header ---
    uint32_t headerBytes = reader.getBytesRead();
    uint8_t* payloadPtr = ptr + headerBytes;

   // printf("Pointer after reading header: %p\n", payloadPtr);
    // Compute number of weights
    uint32_t numWeights = 1;
    for (uint32_t i = 0; i < numDims; i++)
        numWeights *= shape[i];

    // Resize tensor data to hold decoded weights
    tensor.data.resize(numWeights);
   // printf("Decoding tensor: id=%d type=%d bitwidth=%d numDims=%d numWeights=%d\n",
       // tensor.tensorId, static_cast<uint32_t>(tensor.tensorType), static_cast<uint32_t>(tensor.tensorBitwidth), numDims, numWeights);

    //m_BACDecoder.startBacDecoding(payloadPtr); // set BAC decoder to start of payload 

    // Decode weights
    uint8_t* endPtr = m_BACDecoder.decodeWeights(tensor.data.data(), numWeights, payloadPtr); // decodeWeights returns pointer after weights


     return endPtr; // return pointer to next tensor header (or end of stream)
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
void Decoder::decodeModel(std::vector<TensorMeta>& modelTensors, const std::vector<uint8_t>& bytestream)
{
    // Decode number of tensors from raw header (not BAC encoded)
    BitstreamReader reader(bytestream.data() );

    printf("Bytestream pointer in decodeModel: %p\n", bytestream.data());

    uint32_t numTensors = reader.readBits(MAX_TENSORS_BITS); // read numTensors
    //printf("Decoded numTensors: %d\n", numTensors);
    reader.alignToByte(); // align to byte after reading numTensors

    printf("Decoding model with %d tensors\n", numTensors);

    modelTensors.resize(numTensors);

    // pointer after global header
    uint8_t* ptr = const_cast<uint8_t*>(bytestream.data()) + (reader.getBytesRead());
   // printf("Pointer after reading numTensors: %p\n", ptr);
    for (uint32_t i = 0; i < numTensors; i++)
    {
        ptr = decodeLayer(modelTensors[i], ptr);   // fills TensorMeta directly
      //  printf("Pointer after decoding tensor %d: %p\n", i, ptr);
    }
    //finishDecoding();
}

