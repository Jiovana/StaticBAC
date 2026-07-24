#pragma once

#include <array>
#include <cmath>
#include <cstdint>

#include  "../CommonLib/ContextModel.h"
#include "../CommonLib/TypeDef.h"


class BitEstimator
{
public:

    static constexpr int NUM_TYPES = 2;
    static constexpr int NUM_PRED  = 3;
    static constexpr int NUM_CTX   = 7;
    

    // cost[type][predictor][ctx][symbol]
    std::array<
        std::array<
            std::array<
                std::array<uint16_t,2>,
                NUM_CTX>,
            NUM_PRED>,
        NUM_TYPES> cost;

    BitEstimator()
    {
        generateTable();
    }

    void generateTable()
    {
        for(int type = 0; type < NUM_TYPES; type++)
        {
            TensorType tensorType =
                (type == 0) ? TensorType::Weight
                            : TensorType::Bias;

            for(int pred = 0; pred < NUM_PRED; pred++)
            {
                for(int ctx = 0; ctx < NUM_CTX; ctx++)
                {
                    double pLPS =
                        StaticCtx::getRLPS(ctx, tensorType, pred) / StaticCtx::CABAC_RANGE;

                    if(pLPS < 1e-9)
                        pLPS = 1e-9;

                    double pMPS = 1.0 - pLPS;

                    uint8_t mps =
                        StaticCtx::getMPS(ctx, tensorType, pred);

                    uint16_t mpsCost =
                        uint16_t(std::round(-std::log2(pMPS) * 256.0));

                    uint16_t lpsCost =
                        uint16_t(std::round(-std::log2(pLPS) * 256.0));

                    cost[type][pred][ctx][mps]     = mpsCost;
                    cost[type][pred][ctx][1-mps]   = lpsCost;
                }
            }
        }
    }

     void printTable() const
    {
        std::cout << "\n========== BitEstimator Cost Table ==========\n";

        for(int type = 0; type < NUM_TYPES; type++)
        {
            std::cout << "\nTensor type: "
                      << (type == 0 ? "Weights" : "Biases")
                      << "\n";

            for(int pred = 0; pred < NUM_PRED; pred++)
            {
                std::cout << "\nPredictor: ";

                switch(pred)
                {
                    case 0: std::cout << "NONE"; break;
                    case 1: std::cout << "MEAN"; break;
                    case 2: std::cout << "NEIGHBOR"; break;
                }

                std::cout << "\n";
                std::cout << "Ctx\tMPS\tRLPS\tCost(0)\tCost(1)\n";

                for(int ctx = 0; ctx < NUM_CTX; ctx++)
                {
                    TensorType tensor =
                        (type == 0) ? TensorType::Weight
                                    : TensorType::Bias;

                    StaticCtx staticCtx;

                    std::cout
                        << ctx << "\t"
                        << int(staticCtx.getMPS(ctx,tensor,pred)) << "\t"
                        << int(staticCtx.getRLPS(ctx,tensor,pred)) << "\t"
                        << cost[type][pred][ctx][0]/256.0 << "\t"
                        << cost[type][pred][ctx][1]/256.0 << "\n";
                }
            }
        }

        std::cout << "=============================================\n";
    }

    inline double estimate(uint8_t ctx,
                           uint8_t symbol,
                           TensorType type,
                           uint8_t predictor) const
    {
        return cost[(int)type][predictor][ctx][symbol] / 256.0;
    }
};