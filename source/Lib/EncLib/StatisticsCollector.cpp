#include "StatisticsCollector.h"

#include <cmath>
#include <iomanip>

void StatisticsCollector::clear()
{
    *this = StatisticsCollector();
}

void StatisticsCollector::save(const std::string& filename) const
{
    std::ofstream f(filename);

    f << std::fixed << std::setprecision(6);

    for(int type=0; type<NUM_TYPES; type++)
    {
        f << (type==0 ? "WEIGHTS\n" : "BIASES\n");

        for(int pred=0; pred<NUM_PRED; pred++)
        {
            static const char* predName[3]={
                "NONE",
                "MEAN",
                "NEIGHBOR"
            };

            f << "\nPredictor " << predName[pred] << "\n";

            //---------------- SIG ----------------

            f << "SIG\n";

            for(int ctx=0; ctx<NUM_CTX; ctx++)
            {
                auto const& c = m_sig[type][pred][ctx];

                f
                << "ctx "
                << ctx
                << " zeros "
                << c.zero
                << " ones "
                << c.one
                << " p1 "
                << c.probabilityOne()
                << " rlps "
                << c.rlps()
                << "\n";
            }

            //---------------- GT ----------------

            f << "GT\n";

            for(int ctx=0; ctx<NUM_CTX; ctx++)
            {
                auto const& c = m_gt[type][pred][ctx];

                f
                << "ctx "
                << ctx
                << " zeros "
                << c.zero
                << " ones "
                << c.one
                << " p1 "
                << c.probabilityOne()
                << " rlps "
                << c.rlps()
                << "\n";
            }

            f << "\n";
        }
    }
}