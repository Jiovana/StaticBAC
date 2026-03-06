/* -----------------------------------------------------------------------------
The copyright in this software is being made available under the Clear BSD
License, included below. No patent rights, trademark rights and/or
other Intellectual Property Rights other than the copyrights concerning
the Software are granted under this license.

The Clear BSD License

Copyright (c) 2019-2025, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The NNCodec Authors.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted (subject to the limitations in the disclaimer below) provided that
the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

     * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.


------------------------------------------------------------------------------------------- */
#pragma once

#include <cstdint>
#include <vector>
#include <utility>
#include <sstream>
#include <cstddef>
#include <cstring>
#include <assert.h>
#include <cassert>
//#include <pybind11/pybind11.h>

#define CFG_FIX_TC    1

enum class ctxIds
{
  // TBD: Use also other indices
  sigfBaEquZroNbEquZro = 0,
  sigfBaEquZroNbGrtZro = 1,
  sigfBaEquZroNbLesZro = 2,
  sigfHdsp             = 3,
  sigfBaEquOne         = 4,
  sigfBaGrtOne         = 5,
  signBaEquZroNbEquZro = 0 + 8 * 6,
  signBaEquZroNbGrtZro = 1 + 8 * 6,
  signBaEquZroNbLesZro = 2 + 8 * 6,
  signBaEquOne         = 3 + 8 * 6,
  signBaGrtOne         = 4 + 8 * 6,
  gtx0BaEquZroCvGrt0   = 0 + 8 * 6 + 6,
  gtx0BaEquZroCvLeq0   = 1 + 8 * 6 + 6,
  gtx1BaEquZroCvGrt0   = 2 + 8 * 6 + 6,
  gtx1BaEquZroCvLeq0   = 3 + 8 * 6 + 6,
  gtx2BaEquZroCvGrt0   = 4 + 8 * 6 + 6,
  gtx2BaEquZroCvLeq0   = 5 + 8 * 6 + 6,
  gtx3BaEquZroCvGrt0   = 6 + 8 * 6 + 6,
  gtx3BaEquZroCvLeq0   = 7 + 8 * 6 + 6,
  gtx4BaEquZroCvGrt0   = 8 + 8 * 6 + 6,
  gtx4BaEquZroCvLeq0   = 9 + 8 * 6 + 6,
};


enum class TensorType : uint8_t
{
    Weight = 0,
    Bias   = 1
};

enum class TensorBitwidth : uint8_t
{
    BW_4  = 0,
    BW_8  = 1,
    BW_12 = 2,
    BW_16 = 3,
    BW_20 = 4,
    BW_24 = 5,
    BW_32 = 6
};

inline uint32_t getBitwidthFromEnum (TensorBitwidth bw)
{
    uint32_t tensorwidth = 0;
    switch (bw)
    {
        case TensorBitwidth::BW_4:  tensorwidth = 4; break;
        case TensorBitwidth::BW_8:  tensorwidth = 8; break;
        case TensorBitwidth::BW_12: tensorwidth = 12; break;
        case TensorBitwidth::BW_16: tensorwidth = 16; break;
        case TensorBitwidth::BW_20: tensorwidth = 20; break;
        case TensorBitwidth::BW_24: tensorwidth = 24; break;
        case TensorBitwidth::BW_32: tensorwidth = 32; break;
        default:                    tensorwidth = 8; break; // default to 8 if unknown
    }
    return tensorwidth;
};

inline uint8_t getShiftMaxFromBitwidth (TensorBitwidth bw)
{
    uint8_t shiftMax = 0;
    switch (bw)
    {
        case TensorBitwidth::BW_4:  shiftMax = 0; break; // 4 - 2 = 2
        case TensorBitwidth::BW_8:  shiftMax = 2; break; // 8 - 2 = 6
        case TensorBitwidth::BW_12: shiftMax = 5; break; // 12 - 2 = 10
        case TensorBitwidth::BW_16: shiftMax = 6; break; // 16 - 2 = 14
        case TensorBitwidth::BW_20: shiftMax = 7; break; // 20 - 2 = 18
        case TensorBitwidth::BW_24: shiftMax = 8; break; // 24 - 2 = 22
        case TensorBitwidth::BW_32: shiftMax = 9; break; // 32 - 2 = 30
        default:                    shiftMax = 2; break; // default to shift max of 6 for unknown bitwidths
    }
    return shiftMax;
};
inline uint8_t getShiftFromMeanAndK(TensorBitwidth bw, int32_t mean, uint32_t k)
{
    uint8_t shiftMax = getShiftMaxFromBitwidth(bw);

    int absMean = std::abs(mean);
    if (absMean == 0)
        return 0;

    int meanBits = 32 - __builtin_clz(absMean);

    // base shift from mean
    int shift = meanBits - 6;
    if (shift < 0) shift = 0;

    printf("Calculated base shift from mean: %d (mean=%d)\n", shift, mean);
    // scale positively with k
    shift += static_cast<int>(k+1);

    // cap to safe values
    if (shift > shiftMax) shift = shiftMax;

    printf("Final calculated shift after adding k and capping: %d\n", shift);

    return static_cast<uint8_t>(shift);
}

enum class TensorRole {
    UNKNOWN,
    WEIGHT_CONV,       // convolution weights
    WEIGHT_FC,         // fully connected / linear layer
    WEIGHT_ATT_Q,      // attention query
    WEIGHT_ATT_K,      // attention key
    WEIGHT_ATT_V,      // attention value
    WEIGHT_ATT_O,      // attention output
    WEIGHT_EMBEDDING,  // embeddings
    WEIGHT_NORM,       // LayerNorm / BatchNorm
    WEIGHT_CLASSIFIER, // classifier head
    BIAS,              // biases for any layer
    OTHER
};


// for future use
struct TensorMeta
{
    TensorType type;
    TensorBitwidth bitwidth;
    uint32_t numDims;
    const uint32_t* shape;
};

//using namespace pybind11::literals;

class Exception : public std::exception
{
public:
  Exception( const std::string& _s ) : m_str( _s ) { }
  Exception( const Exception& _e ) : std::exception( _e ), m_str( _e.m_str ) { }
  virtual ~Exception() noexcept { };
  virtual const char* what() const noexcept { return m_str.c_str(); }
  Exception& operator=( const Exception& _e ) { std::exception::operator=( _e ); m_str = _e.m_str; return *this; }
  template<typename T> Exception& operator<<( T t ) { std::ostringstream oss; oss << t; m_str += oss.str(); return *this; }
private:
  std::string m_str;
};

#define THROW(x)            throw( Exception( "\nERROR: In function \"" ) << __FUNCTION__ << "\" in " << __FILE__ << ":" << __LINE__ << ": " << x )
#define CHECK(c,x)          if(c){ THROW(x); }

typedef float float32_t;

