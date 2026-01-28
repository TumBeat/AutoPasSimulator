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

#include <utils/KokkosParticle.h>

#include "Configuration.h"

namespace utils {

    class Setup {
    public:
        template <class Container>
        void static provideOptions(Container& autopasInstance, const Configuration& config) {
            autopasInstance.setAllowedContainers({autopas::options::ContainerOption::kokkosDirectSum});
            autopasInstance.setAllowedDataLayouts({autopas::options::DataLayoutOption::soa});
            autopasInstance.setAllowedContainerLayouts({autopas::options::DataLayoutOption::soa});
            autopasInstance.setAllowedNewton3Options({autopas::options::Newton3Option::enabled});
            autopasInstance.setAllowedInteractionTypeOptions({autopas::InteractionTypeOption::pairwise});

            autopasInstance.setCutoff(config.getCutoff());
            autopasInstance.setBoxMin({config.getBoxMin(), config.getBoxMin(), config.getBoxMin()});
            autopasInstance.setBoxMax({config.getBoxMax(), config.getBoxMax(), config.getBoxMax()});
        }

        template <class Container>
        void static fillParticles(Container& autopasInstance, const Configuration& config) {

            std::uniform_real_distribution<KokkosParticle::ParticleSoAFloatPrecision> distribution(config.getBoxMin(),config.getBoxMax());
            std::uniform_real_distribution<KokkosParticle::ParticleSoAFloatPrecision> haloDistribution(config.getBoxMax() + 0.1 ,config.getBoxMax() + config.getCutoff());
            std::default_random_engine generator;

            for (int i = 0; i < config.getNumParticles(); i++) {
                KokkosParticle p {};
                p.setF({0.,0.,0.});
                p.setR({distribution(generator), distribution(generator), distribution(generator)});
                p.setID(i);
                p.setMass(1.);

                autopasInstance.addParticle(p);
            }

            for (int i = 0; i < config.getNumHalos(); i++) {
                KokkosParticle p {};
                p.setF({0.,0.,0.});
                p.setR({haloDistribution(generator), distribution(generator), distribution(generator)});
                p.setID(config.getNumParticles() + i);
                p.setMass(1.);

                autopasInstance.addHaloParticle(p);
            }
        }
    };

}
