/**
 * @file FunctorKokkos.h
 * @date 11.12.2025
 * @author Luis Gall
 */

#pragma once

#include "autopas/baseFunctors/PairwiseFunctor.h"
#include "autopas/utils/SoAView.h"

template <class Particle_T, class MemSpace>
class FunctorKokkos : public autopas::PairwiseFunctor<Particle_T, FunctorKokkos<Particle_T, MemSpace>, MemSpace> {

public:
    using SoAArraysType = typename Particle_T::SoAArraysType;

    explicit FunctorKokkos(double cutoff)
        : autopas::PairwiseFunctor<Particle_T, FunctorKokkos<Particle_T, MemSpace>, MemSpace>(cutoff),
        _cutoffSquared{cutoff * cutoff}
    {}

    /* Overrides for actual execution */
    void AoSFunctor(Particle_T& i, Particle_T& j, bool newton3) final {

    }

    void SoAFunctorSingle(autopas::SoAView<SoAArraysType> soa, bool newton3) final {
        // No-op as nothing should happen here
    }

    void SoAFunctorPair(autopas::SoAView<SoAArraysType> soa1, autopas::SoAView<SoAArraysType> soa2, bool newton3) final {
        // No-op as nothing should happen here
    }

    KOKKOS_INLINE_FUNCTION
    void SoAFunctorSingleKokkos(int i, const Particle_T::KokkosSoAArraysType& soa, bool newton3) final {
        // TODO: size() is not (yet) KOKKOS_INLINE_FUNCTION
        for (int j = 0; j < soa.size() && j != i; ++j) {
            // This is oriented on the SoAKernel of autopas' LJFunctorAVX
            const auto x1 = soa.template operator()<ParticleType::AttributeNames::posX, true, false>(i);
            const auto y1 = soa.template operator()<ParticleType::AttributeNames::posY, true, false>(i);
            const auto z1 = soa.template operator()<ParticleType::AttributeNames::posZ, true, false>(i);

            const auto x2 = soa.template operator()<ParticleType::AttributeNames::posX, true, false>(j);
            const auto y2 = soa.template operator()<ParticleType::AttributeNames::posY, true, false>(j);
            const auto z2 = soa.template operator()<ParticleType::AttributeNames::posZ, true, false>(j);

            const auto drX = x1 - x2;
            const auto drY = y1 - y2;
            const auto drZ = z1 - z2;

            const auto drX2 = drX * drX;
            const auto drY2 = drY * drY;
            const auto drZ2 = drZ * drZ;

            const auto dr2 = drX2 + drY2 + drZ2;

            if (dr2 > _cutoffSquared) {
                return;
            }

            // TODO: mixing calculations goes here

            // TODO: LJ force calculation goes here
        }
    }

    KOKKOS_INLINE_FUNCTION
    void SoAFunctorPairKokkos(Particle_T::KokkosSoAArraysType& soa1, Particle_T::KokkosSoAArraysType& soa2, bool newton3) final {
        // TODO: implement with operator() semantics
    }


    /* Interface required stuff */
    std::string getName() final {
        return "FunctorKokkos";
    }

    bool isRelevantForTuning() final {
        return true;
    }

    bool allowsNewton3() final {
        // TODO
        return true;
    }

    bool allowsNonNewton3() final {
        // TODO
        return true;
    }

private:

    double _cutoffSquared;
};