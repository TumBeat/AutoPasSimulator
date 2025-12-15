/**
 *@file Setup.h
 *@date 13.12.2025
 *@author Luis Gall
 */

#pragma once

#include "autopas/options/ContainerOption.h"
#include "autopas/options/DataLayoutOption.h"
#include "autopas/options/InteractionTypeOption.h"
#include "autopas/options/Newton3Option.h"

#include <utils/ParticleType.h>

namespace utils {

    class Setup {
    public:
        template <class Container>
        void static provideOptions(Container& autopasInstance) {
            autopasInstance.setAllowedContainers({autopas::options::ContainerOption::kokkosDirectSum});
            autopasInstance.setAllowedDataLayouts({autopas::options::DataLayoutOption::soa});
            autopasInstance.setAllowedContainerLayouts({autopas::options::DataLayoutOption::soa});
            autopasInstance.setAllowedNewton3Options({autopas::options::Newton3Option::enabled});
            autopasInstance.setAllowedInteractionTypeOptions({autopas::InteractionTypeOption::pairwise});

            autopasInstance.setCutoff(3.);
            autopasInstance.setBoxMin({-3.,-3.,-3.});
            autopasInstance.setBoxMax({9.,9.,9.});
        }

        template <class Container>
        void static fillParticles(Container& autopasInstance, size_t N) {

            std::uniform_real_distribution<double> distribution(0., 5.);
            std::default_random_engine generator;

            for (int i = 0; i < N; i++) {
                ParticleType p {};
                p.setF({0.,0.,0.});
                p.setR({distribution(generator), distribution(generator), distribution(generator)});
                p.setID(i);

                autopasInstance.addParticle(p);
            }
        }
    };

}
