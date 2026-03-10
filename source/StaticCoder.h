#include "Lib/CommonLib/TypeDef.h"
#include "Lib/EncLib/CABACEncoder.h"
#include "Lib/DecLib/CABACDecoder.h"
#include <iostream>
#include <math.h>


class Encoder
{
public:
  Encoder() { m_CABACEncoder.startCabacEncoding( &m_Bytestream ); }
  ~Encoder() {}
  void                  initCtxModels(uint32_t cabac_unary_length) { m_CABACEncoder.initCtxMdls(cabac_unary_length); }
  void                  iae_v( uint8_t v, int32_t value )            { m_CABACEncoder.iae_v( v, value ); }
  void                  uae_v( uint8_t v, uint32_t value )           { m_CABACEncoder.uae_v( v, value ); }
  uint64_t              encodeLayer(const TensorMeta& tensor, uint16_t tensorId, uint32_t& headerBits);
  const std::vector<uint8_t>&  finishEncoding();
  const std::vector<uint8_t>&  encodeModel(const std::vector<TensorMeta>& modelTensors);
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

  void     decodeLayer  ( TensorMeta& tensor);
  uint32_t finishDecoding       ();
  void     decodeModel(std::vector<TensorMeta>& modelTensors);

  

private:
  CABACDecoder  m_CABACDecoder;
};

