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

    KOKKOS_INLINE_FUNCTION
    void SoAFunctorSingleKokkos(Particle_T::template KokkosSoAArraysType<MemSpace>& soa, bool newton3) final {
        // TODO: implement with operator() semantics
    }

    KOKKOS_INLINE_FUNCTION
    void SoAFunctorPairKokkos(Particle_T::template KokkosSoAArraysType<MemSpace>& soa1, Particle_T::template KokkosSoAArraysType<MemSpace>& soa2, bool newton3) final {
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