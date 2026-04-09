#include "..\CommonLib\TypeDef.h"

#ifndef OP_COUNTER_H
#define OP_COUNTER_H

#pragma once

struct OpCounter {
    uint64_t add = 0;
    uint64_t sub = 0;
    uint64_t mul = 0;
    uint64_t shift = 0;
    uint64_t cmp = 0;
    uint64_t branch = 0;
    uint64_t mem = 0;

    uint64_t regularBins=0;
    uint64_t bypassBins=0;
    uint64_t loops=0;
};
extern OpCounter g_ops;



#define OP_ADD(x)   (g_ops.add++, (x))
#define OP_SUB(x)   (g_ops.sub++, (x))
#define OP_MUL(x)   (g_ops.mul++, (x))

#define OP_SHL(x,n) (g_ops.shift++, ((x) << (n)))
#define OP_SHR(x,n) (g_ops.shift++, ((x) >> (n)))

#define OP_CMP(x)   (g_ops.cmp++, (x))
#define OP_BRANCH() (g_ops.branch++)

#define OP_MEM()    (g_ops.mem++)

#endif // OP_COUNTER_H