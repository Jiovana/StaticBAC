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
#ifndef __CONTEXTMODEL__
#define __CONTEXTMODEL__

#include "TypeDef.h"

namespace{

    constexpr uint8_t NUM_CTX  = 7;
    constexpr uint8_t NUM_PRED = 3;

    constexpr uint8_t rlpsTable[2][NUM_PRED][NUM_CTX] =
        {
            // WEIGHTS
            {
                // NONE
                {81,25,99,52,227,110,25},
                // MEAN
                {25,15,75,43,249,117,50},
                // NEIGHBOR
                {67,28,121,61,222,126,43}
            },
            // BIASES
            {
                // NONE
                {190,26,20,8,66,116,161},
                // MEAN
                {8,9,62,22,56,196,203},
                // NEIGHBOR
                {142,31,170,10,55,105,151}
            }
        };

        // Weight MPS is identical for every predictor
        constexpr uint8_t weightMps[NUM_CTX] =
        {
            1,1,1,1,0,0,0
        };

        // Bias MPS
        constexpr uint8_t biasMps[NUM_PRED][NUM_CTX] =
        {
            // NONE
            {0,1,1,1,1,1,1},
            // MEAN
            {1,1,1,1,1,1,0},
            // NEIGHBOR
            {1,1,1,1,1,1,1}
        };
}

class StaticCtx
{
public:

    static constexpr double CABAC_RANGE = 512.0;

    static uint8_t getRLPS( uint8_t ctxId,  TensorType paramType, uint8_t pred ) ;
    static uint8_t getMPS( uint8_t ctxId, TensorType paramType, uint8_t pred ) ;




};

#endif // __CONTEXTMODEL__
