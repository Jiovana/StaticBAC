#pragma once
#include <cstdint>

struct StaticCtx
{
    // fixed probability: P(LPS) = 1 / 16
    inline uint32_t getRLPS(uint32_t range) const
    {
        return range >> 4;
    }

    inline int getMinusMPS() const
    {
        return 0;
    }

    inline void updateState(int) const
    {
        // static: do nothing
    }
};
