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
#include "ContextModel.h"
#include <iostream>

const uint8_t rps_table[2][13] = {
  {44, 65, 37, 223, 59, 105, 10, 26, 41, 56, 72, 87, 154},
  {27, 10, 38, 209, 42, 38, 15, 46, 82, 128, 154, 179, 102}
};

const uint8_t mps_table[2][13] = {
  {1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1},
  {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1}
};


uint8_t StaticCtx::getRLPS( uint8_t ctxId,  TensorType paramType ) const
{
  //PROFILE_SCOPE("StaticCtx_getRLPS", 0);
    //std::cout << "getRLPS called with ctxId: " << (int)ctxId << " paramType: " << (int)paramType << std::endl;
    int type = paramType == TensorType::Weight ? 0 : 1; // example mapping, adjust as needed
    return rps_table[type][ctxId];
}

uint8_t StaticCtx::getMPS( uint8_t ctxId, TensorType paramType ) const
{
  int type = paramType == TensorType::Weight ? 0 : 1; // example mapping, adjust as needed
  return mps_table[type][ctxId];
}
