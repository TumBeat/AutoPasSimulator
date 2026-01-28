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

    void SoAFunctorSingleKokkos(const Particle_T::KokkosSoAArraysType& soa, bool newton3) final {

        const size_t N = soa.size();
        const typename Particle_T::ParticleSoAFloatPrecision cutoffSquared = static_cast<typename Particle_T::ParticleSoAFloatPrecision>(_cutoffSquared);

        //Kokkos::parallel_for(Kokkos::TeamPolicy<typename MemSpace::execution_space>(N, Kokkos::AUTO()), KOKKOS_LAMBDA(Kokkos::TeamPolicy<typename MemSpace::execution_space>::member_type team) {
        Kokkos::parallel_for(Kokkos::RangePolicy<typename MemSpace::execution_space>(0, N), KOKKOS_LAMBDA(int i) {
            //int i = team.league_rank();

            const auto owned1 = soa.template operator()<Particle_T::AttributeNames::ownershipState, true, false>(i);

            if (owned1 != autopas::OwnershipState::dummy) {
                typename Particle_T::ParticleSoAFloatPrecision fxAcc = 0.;
                typename Particle_T::ParticleSoAFloatPrecision fyAcc = 0.;
                typename Particle_T::ParticleSoAFloatPrecision fzAcc = 0.;

                const typename Particle_T::ParticleSoAFloatPrecision x1 = soa.template operator()<Particle_T::AttributeNames::posX, true, false>(i);
                const typename Particle_T::ParticleSoAFloatPrecision y1 = soa.template operator()<Particle_T::AttributeNames::posY, true, false>(i);
                const typename Particle_T::ParticleSoAFloatPrecision z1 = soa.template operator()<Particle_T::AttributeNames::posZ, true, false>(i);

                for (int j = 0; j < N; ++j) {
                    if (i != j) {

                        const auto owned2 = soa.template operator()<Particle_T::AttributeNames::ownershipState, true, false>(j);

                        if (owned2 != autopas::OwnershipState::dummy) {
                            const typename Particle_T::ParticleSoAFloatPrecision x2 = soa.template operator()<Particle_T::AttributeNames::posX, true, false>(j);
                            const typename Particle_T::ParticleSoAFloatPrecision y2 = soa.template operator()<Particle_T::AttributeNames::posY, true, false>(j);
                            const typename Particle_T::ParticleSoAFloatPrecision z2 = soa.template operator()<Particle_T::AttributeNames::posZ, true, false>(j);

                            const typename Particle_T::ParticleSoAFloatPrecision drX = x1 - x2;
                            const typename Particle_T::ParticleSoAFloatPrecision drY = y1 - y2;
                            const typename Particle_T::ParticleSoAFloatPrecision drZ = z1 - z2;

                            const typename Particle_T::ParticleSoAFloatPrecision drX2 = drX * drX;
                            const typename Particle_T::ParticleSoAFloatPrecision drY2 = drY * drY;
                            const typename Particle_T::ParticleSoAFloatPrecision drZ2 = drZ * drZ;

                            const typename Particle_T::ParticleSoAFloatPrecision dr2 = drX2 + drY2 + drZ2;

                            if (dr2 <= cutoffSquared) {
                                // TODO: consider mixing based on type or some sort of parameter injection
                                const typename Particle_T::ParticleSoAFloatPrecision sigmaSquared = 1.;
                                const typename Particle_T::ParticleSoAFloatPrecision epsilon24 = 24.;

                                const typename Particle_T::ParticleSoAFloatPrecision invDr2 = 1.0 / dr2;
                                typename Particle_T::ParticleSoAFloatPrecision lj6 = sigmaSquared * invDr2;
                                lj6 = lj6 * lj6 * lj6;
                                const typename Particle_T::ParticleSoAFloatPrecision lj12 = lj6 * lj6;
                                const typename Particle_T::ParticleSoAFloatPrecision lj12m6 = lj12 - lj6;
                                const typename Particle_T::ParticleSoAFloatPrecision fac = epsilon24 * (lj12 + lj12m6) * invDr2;

                                const typename Particle_T::ParticleSoAFloatPrecision fX = fac * drX;
                                const typename Particle_T::ParticleSoAFloatPrecision fY = fac * drY;
                                const typename Particle_T::ParticleSoAFloatPrecision fZ = fac * drZ;

                                // TODO: consider newton3 if enabled
                                fxAcc += fX;
                                fxAcc += fY;
                                fzAcc += fZ;
                            }
                        }
                    }
                }
                //}, fxAcc, fyAcc, fzAcc);
                //team.team_barrier();

                const typename Particle_T::ParticleSoAFloatPrecision oldFx = soa.template operator()<Particle_T::AttributeNames::forceX, true, false>(i);
                const typename Particle_T::ParticleSoAFloatPrecision oldFy = soa.template operator()<Particle_T::AttributeNames::forceY, true, false>(i);
                const typename Particle_T::ParticleSoAFloatPrecision oldFz = soa.template operator()<Particle_T::AttributeNames::forceZ, true, false>(i);

                const typename Particle_T::ParticleSoAFloatPrecision newFx = oldFx + fxAcc;
                const typename Particle_T::ParticleSoAFloatPrecision newFy = oldFy + fyAcc;
                const typename Particle_T::ParticleSoAFloatPrecision newFz = oldFz + fzAcc;

                soa.template operator()<Particle_T::AttributeNames::forceX, true, false>(i) = newFx;
                soa.template operator()<Particle_T::AttributeNames::forceY, true, false>(i) = newFy;
                soa.template operator()<Particle_T::AttributeNames::forceZ, true, false>(i) = newFz;
            }
        });
    }

    void SoAFunctorPairKokkos(const Particle_T::KokkosSoAArraysType& soa1, const Particle_T::KokkosSoAArraysType& soa2, bool newton3) final {
        const size_t N = soa1.size();
        const size_t M = soa2.size();

        const typename Particle_T::ParticleSoAFloatPrecision cutoffSquared = static_cast<typename Particle_T::ParticleSoAFloatPrecision>(_cutoffSquared);

        //Kokkos::TeamPolicy<typename MemSpace::execution_space> policy (N, Kokkos::AUTO());

        Kokkos::parallel_for(Kokkos::RangePolicy<typename MemSpace::execution_space>(0, N), KOKKOS_LAMBDA(int i) {
            //int i = team.league_rank();

            const auto owned1 = soa1.template operator()<Particle_T::AttributeNames::ownershipState, true, false>(i);

            if (owned1 != autopas::OwnershipState::dummy) {
                typename Particle_T::ParticleSoAFloatPrecision fxAcc = 0.;
                typename Particle_T::ParticleSoAFloatPrecision fyAcc = 0.;
                typename Particle_T::ParticleSoAFloatPrecision fzAcc = 0.;

                const typename Particle_T::ParticleSoAFloatPrecision x1 = soa1.template operator()<Particle_T::AttributeNames::posX, true, false>(i);
                const typename Particle_T::ParticleSoAFloatPrecision y1 = soa1.template operator()<Particle_T::AttributeNames::posY, true, false>(i);
                const typename Particle_T::ParticleSoAFloatPrecision z1 = soa1.template operator()<Particle_T::AttributeNames::posZ, true, false>(i);

                for (int j = 0; j < M; ++j) {
                    const auto owned2 = soa2.template operator()<Particle_T::AttributeNames::ownershipState, true, false>(j);

                    if (owned2 != autopas::OwnershipState::dummy) {
                        const typename Particle_T::ParticleSoAFloatPrecision x2 = soa2.template operator()<Particle_T::AttributeNames::posX, true, false>(j);
                        const typename Particle_T::ParticleSoAFloatPrecision y2 = soa2.template operator()<Particle_T::AttributeNames::posY, true, false>(j);
                        const typename Particle_T::ParticleSoAFloatPrecision z2 = soa2.template operator()<Particle_T::AttributeNames::posZ, true, false>(j);

                        const typename Particle_T::ParticleSoAFloatPrecision drX = x1 - x2;
                        const typename Particle_T::ParticleSoAFloatPrecision drY = y1 - y2;
                        const typename Particle_T::ParticleSoAFloatPrecision drZ = z1 - z2;

                        const typename Particle_T::ParticleSoAFloatPrecision drX2 = drX * drX;
                        const typename Particle_T::ParticleSoAFloatPrecision drY2 = drY * drY;
                        const typename Particle_T::ParticleSoAFloatPrecision drZ2 = drZ * drZ;

                        const typename Particle_T::ParticleSoAFloatPrecision dr2 = drX2 + drY2 + drZ2;

                        if (dr2 <= cutoffSquared) {
                            // TODO: consider mixing based on type or some sort of parameter injection
                            const typename Particle_T::ParticleSoAFloatPrecision sigmaSquared = 1.;
                            const typename Particle_T::ParticleSoAFloatPrecision epsilon24 = 24.;

                            const typename Particle_T::ParticleSoAFloatPrecision invDr2 = 1.0 / dr2;
                            typename Particle_T::ParticleSoAFloatPrecision lj6 = sigmaSquared * invDr2;
                            lj6 = lj6 * lj6 * lj6;
                            const typename Particle_T::ParticleSoAFloatPrecision lj12 = lj6 * lj6;
                            const typename Particle_T::ParticleSoAFloatPrecision lj12m6 = lj12 - lj6;
                            const typename Particle_T::ParticleSoAFloatPrecision fac = epsilon24 * (lj12 + lj12m6) * invDr2;

                            const typename Particle_T::ParticleSoAFloatPrecision fX = fac * drX;
                            const typename Particle_T::ParticleSoAFloatPrecision fY = fac * drY;
                            const typename Particle_T::ParticleSoAFloatPrecision fZ = fac * drZ;

                            // TODO: consider newton3 if enabled
                            fxAcc += fX;
                            fyAcc += fY;
                            fzAcc += fZ;
                        }
                    }
                //}, fxAcc, fyAcc, fzAcc);
                }

                const typename Particle_T::ParticleSoAFloatPrecision oldFx = soa1.template operator()<Particle_T::AttributeNames::forceX, true, false>(i);
                const typename Particle_T::ParticleSoAFloatPrecision oldFy = soa1.template operator()<Particle_T::AttributeNames::forceY, true, false>(i);
                const typename Particle_T::ParticleSoAFloatPrecision oldFz = soa1.template operator()<Particle_T::AttributeNames::forceZ, true, false>(i);

                const typename Particle_T::ParticleSoAFloatPrecision newFx = oldFx + fxAcc;
                const typename Particle_T::ParticleSoAFloatPrecision newFy = oldFy + fyAcc;
                const typename Particle_T::ParticleSoAFloatPrecision newFz = oldFz + fzAcc;

                soa1.template operator()<Particle_T::AttributeNames::forceX, true, false>(i) = newFx;
                soa1.template operator()<Particle_T::AttributeNames::forceY, true, false>(i) = newFy;
                soa1.template operator()<Particle_T::AttributeNames::forceZ, true, false>(i) = newFz;
            }
        });
    }

    constexpr static auto getNeededAttr() {
        return std::array<typename Particle_T::AttributeNames, 9>{
            Particle_T::AttributeNames::id,
            Particle_T::AttributeNames::posX,
            Particle_T::AttributeNames::posY,
            Particle_T::AttributeNames::posZ,
            Particle_T::AttributeNames::forceX,
            Particle_T::AttributeNames::forceY,
            Particle_T::AttributeNames::forceZ,
            Particle_T::AttributeNames::typeId,
            Particle_T::AttributeNames::ownershipState,
        };
    }

    constexpr static auto getNeededAttr(std::false_type) {
        return std::array<typename Particle_T::AttributeNames, 6>{
            Particle_T::AttributeNames::id,
            Particle_T::AttributeNames::posX,
            Particle_T::AttributeNames::posY,
            Particle_T::AttributeNames::posZ,
            Particle_T::AttributeNames::typeId,
            Particle_T::AttributeNames::ownershipState};
    }

    constexpr static auto getComputedAttr() {
        return std::array<typename Particle_T::AttributeNames, 3>{
            Particle_T::AttributeNames::forceX,
            Particle_T::AttributeNames::forceY,
            Particle_T::AttributeNames::forceZ
        };
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