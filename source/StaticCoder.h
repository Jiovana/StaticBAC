#include "Lib/CommonLib/TypeDef.h"
#include "Lib/EncLib/CABACEncoder.h"
#include "Lib/DecLib/CABACDecoder.h"
#include <iostream>
#include <math.h>

struct TensorMeta
{
    std::string name;                 // Tensor name (e.g., "encoder_layer_0_weight")
    std::vector<int32_t> data;        // Tensor data (weights or biases)
    std::vector<uint32_t> shape;      // Shape of the tensor (1D, 2D, etc.)
    uint32_t numDims;                 // Number of dimensions
    TensorType TensorType;            // weight, bias, other
    TensorBitwidth tensorBitwidth;     //b4, b8, b12, b16..

    TensorMeta() : numDims(0) {}
    TensorMeta(const std::string& n, const std::vector<int32_t>& d, const std::vector<uint32_t>& s)
        : name(n), data(d), shape(s), numDims(static_cast<uint32_t>(s.size())) {}
};

class Encoder
{
public:
  Encoder() { m_CABACEncoder.startCabacEncoding( &m_Bytestream ); }
  ~Encoder() {}
  void                  initCtxModels(uint32_t cabac_unary_length) { m_CABACEncoder.initCtxMdls(cabac_unary_length); }
  void                  iae_v( uint8_t v, int32_t value )            { m_CABACEncoder.iae_v( v, value ); }
  void                  uae_v( uint8_t v, uint32_t value )           { m_CABACEncoder.uae_v( v, value ); }
  uint64_t Encoder::encodeLayer(const TensorMeta& tensor);
  const std::vector<uint8_t>&  finishEncoding();
  const std::vector<uint8_t>& Encoder::encodeModel(const std::vector<TensorMeta>& modelTensors);
 private:
  std::vector<uint8_t>  m_Bytestream;
  CABACEncoder          m_CABACEncoder;
};


class Decoder
{
public:
  Decoder(){}
  ~Decoder() {}

  void     setStream    ( std::vector<uint8_t>& Bytestream );
  void     initCtxModels( uint32_t cabac_unary_length ) { m_CABACDecoder.initCtxModels( cabac_unary_length ); }
  int32_t  iae_v        (uint8_t v)                     { return m_CABACDecoder.iae_v(v); }
  uint32_t uae_v        ( uint8_t v )                   { return m_CABACDecoder.uae_v( v ); }

  void     decodeLayer  ( std::vector<int32_t>& Weights);
  uint32_t finishDecoding       ();
  void Decoder::decodeModel(std::vector<TensorMeta>& modelTensors);

  

private:
  CABACDecoder  m_CABACDecoder;
};

