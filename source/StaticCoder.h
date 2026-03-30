#include "Lib/CommonLib/TypeDef.h"
#include "Lib/EncLib/CABACEncoder.h"
#include "Lib/DecLib/CABACDecoder.h"
#include <iostream>
#include <math.h>


class Encoder
{
public:
  Encoder() { m_BACEncoder.startBacEncoding( &m_Bytestream ); }
  ~Encoder() {}
  void                  initCtxModels(uint32_t cabac_unary_length) { m_BACEncoder.initCtxMdls(cabac_unary_length); }
  void                  iae_v( uint8_t v, int32_t value )            { m_BACEncoder.iae_v( v, value ); }
  void                  uae_v( uint8_t v, uint32_t value )           { m_BACEncoder.uae_v( v, value ); }
  uint64_t              encodeLayer(const TensorMeta& tensor, uint16_t tensorId, uint32_t& headerBits);
  const std::vector<uint8_t>&  finishEncoding();
  const std::vector<uint8_t>&  encodeModel(const std::vector<TensorMeta>& modelTensors);
 private:
  std::vector<uint8_t>  m_Bytestream;
  BACEncoder          m_BACEncoder;
};


class Decoder
{
public:
  Decoder(){}
  ~Decoder() {}

  void     setStream    ( std::vector<uint8_t>& Bytestream );
  void     initCtxModels( uint32_t cabac_unary_length ) { m_BACDecoder.initCtxModels( cabac_unary_length ); }
  int32_t  iae_v        (uint8_t v)                     { return m_BACDecoder.iae_v(v); }
  uint32_t uae_v        ( uint8_t v )                   { return m_BACDecoder.uae_v( v ); }

  void     decodeLayer  ( TensorMeta& tensor);
  uint32_t finishDecoding       ();
  void     decodeModel(std::vector<TensorMeta>& modelTensors);

  

private:
  BACDecoder  m_BACDecoder;
};

