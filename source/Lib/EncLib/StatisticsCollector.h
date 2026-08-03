#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <string>

class StatisticsCollector
{
public:

    static constexpr int NUM_TYPES = 2;
    static constexpr int NUM_PRED  = 3;
    static constexpr int NUM_CTX   = 7;

    struct BinCounter
    {
        uint64_t zero = 0;
        uint64_t one  = 0;

        inline void update(bool bin)
        {
            if(bin)
                one++;
            else
                zero++;
        }

        inline uint64_t total() const
        {
            return zero + one;
        }

        inline double probabilityOne() const
        {
            uint64_t t = total();
            if(t == 0)
                return 0.5;

            return double(one) / double(t);
        }

        inline uint16_t rlps() const
        {
            double p = probabilityOne();

            if(p > 0.5)
                p = 1.0 - p;

            return uint16_t(std::lround(p * 512.0));
        }
    };

private:

    // [weight/bias][predictor][context]
    BinCounter m_sig[NUM_TYPES][NUM_PRED][NUM_CTX];

    // GT uses exactly the same contexts
    BinCounter m_gt[NUM_TYPES][NUM_PRED][NUM_CTX];

public:

    void clear();

    inline void collectSig(
        int tensorType,
        int predictor,
        int ctx,
        bool bin)
    {
        m_sig[tensorType][predictor][ctx].update(bin);
    }

    inline void collectGT(
        int tensorType,
        int predictor,
        int ctx,
        bool bin)
    {
        m_gt[tensorType][predictor][ctx].update(bin);
    }

    void save(const std::string& filename) const;
};