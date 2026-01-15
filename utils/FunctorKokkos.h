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
        const double cutoffSquared = _cutoffSquared;

        //Kokkos::parallel_for(Kokkos::TeamPolicy<typename MemSpace::execution_space>(N, Kokkos::AUTO()), KOKKOS_LAMBDA(Kokkos::TeamPolicy<typename MemSpace::execution_space>::member_type team) {
        Kokkos::parallel_for(Kokkos::RangePolicy<typename MemSpace::execution_space>(0, N), KOKKOS_LAMBDA(int i) {
            //int i = team.league_rank();

            const auto owned1 = soa.template operator()<Particle_T::AttributeNames::ownershipState, true, false>(i);

            if (owned1 != autopas::OwnershipState::dummy) {
                double fxAcc = 0.;
                double fyAcc = 0.;
                double fzAcc = 0.;

                const auto x1 = soa.template operator()<Particle_T::AttributeNames::posX, true, false>(i);
                const auto y1 = soa.template operator()<Particle_T::AttributeNames::posY, true, false>(i);
                const auto z1 = soa.template operator()<Particle_T::AttributeNames::posZ, true, false>(i);

                //Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, N), [=](int j, double& localFx, double& localFy, double& localFz) {
                for (int j = 0; j < N; ++j) {
                    if (i != j) {

                        const auto owned2 = soa.template operator()<Particle_T::AttributeNames::ownershipState, true, false>(j);

                        if (owned2 != autopas::OwnershipState::dummy) {
                            const double x2 = soa.template operator()<Particle_T::AttributeNames::posX, true, false>(j);
                            const double y2 = soa.template operator()<Particle_T::AttributeNames::posY, true, false>(j);
                            const double z2 = soa.template operator()<Particle_T::AttributeNames::posZ, true, false>(j);

                            const double drX = x1 - x2;
                            const double drY = y1 - y2;
                            const double drZ = z1 - z2;

                            const double drX2 = drX * drX;
                            const double drY2 = drY * drY;
                            const double drZ2 = drZ * drZ;

                            const double dr2 = drX2 + drY2 + drZ2;

                            if (dr2 <= cutoffSquared) {
                                // TODO: consider mixing based on type or some sort of parameter injection
                                const double sigmaSquared = 1.;
                                const double epsilon24 = 24.;

                                const double invDr2 = 1.0 / dr2;
                                double lj6 = sigmaSquared * invDr2;
                                lj6 = lj6 * lj6 * lj6;
                                const double lj12 = lj6 * lj6;
                                const double lj12m6 = lj12 - lj6;
                                const double fac = epsilon24 * (lj12 + lj12m6) * invDr2;

                                const double fX = fac * drX;
                                const double fY = fac * drY;
                                const double fZ = fac * drZ;

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

                const double oldFx = soa.template operator()<Particle_T::AttributeNames::forceX, true, false>(i);
                const double oldFy = soa.template operator()<Particle_T::AttributeNames::forceY, true, false>(i);
                const double oldFz = soa.template operator()<Particle_T::AttributeNames::forceZ, true, false>(i);

                const double newFx = oldFx + fxAcc;
                const double newFy = oldFy + fyAcc;
                const double newFz = oldFz + fzAcc;

                soa.template operator()<Particle_T::AttributeNames::forceX, true, false>(i) = newFx;
                soa.template operator()<Particle_T::AttributeNames::forceY, true, false>(i) = newFy;
                soa.template operator()<Particle_T::AttributeNames::forceZ, true, false>(i) = newFz;
            }
        });
    }

    void SoAFunctorPairKokkos(const Particle_T::KokkosSoAArraysType& soa1, const Particle_T::KokkosSoAArraysType& soa2, bool newton3) final {
        const size_t N = soa1.size();
        const size_t M = soa2.size();

        const double cutoffSquared = _cutoffSquared;

        //Kokkos::TeamPolicy<typename MemSpace::execution_space> policy (N, Kokkos::AUTO());

        Kokkos::parallel_for(Kokkos::RangePolicy<typename MemSpace::execution_space>(0, N), KOKKOS_LAMBDA(int i) {
            //int i = team.league_rank();

            const auto owned1 = soa1.template operator()<Particle_T::AttributeNames::ownershipState, true, false>(i);

            if (owned1 != autopas::OwnershipState::dummy) {
                double fxAcc = 0.;
                double fyAcc = 0.;
                double fzAcc = 0.;

                const auto x1 = soa1.template operator()<Particle_T::AttributeNames::posX, true, false>(i);
                const auto y1 = soa1.template operator()<Particle_T::AttributeNames::posY, true, false>(i);
                const auto z1 = soa1.template operator()<Particle_T::AttributeNames::posZ, true, false>(i);

                //Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,0,  M), [=](int j, double& localFx, double& localFy, double& localFz) {
                for (int j = 0; j < M; ++j) {
                    const auto owned2 = soa2.template operator()<Particle_T::AttributeNames::ownershipState, true, false>(j);

                    if (owned2 != autopas::OwnershipState::dummy) {
                        const double x2 = soa2.template operator()<Particle_T::AttributeNames::posX, true, false>(j);
                        const double y2 = soa2.template operator()<Particle_T::AttributeNames::posY, true, false>(j);
                        const double z2 = soa2.template operator()<Particle_T::AttributeNames::posZ, true, false>(j);

                        const double drX = x1 - x2;
                        const double drY = y1 - y2;
                        const double drZ = z1 - z2;

                        const double drX2 = drX * drX;
                        const double drY2 = drY * drY;
                        const double drZ2 = drZ * drZ;

                        const double dr2 = drX2 + drY2 + drZ2;

                        if (dr2 <= cutoffSquared) {
                            // TODO: consider mixing based on type or some sort of parameter injection
                            const double sigmaSquared = 1.;
                            const double epsilon24 = 24.;

                            const double invDr2 = 1.0 / dr2;
                            double lj6 = sigmaSquared * invDr2;
                            lj6 = lj6 * lj6 * lj6;
                            const double lj12 = lj6 * lj6;
                            const double lj12m6 = lj12 - lj6;
                            const double fac = epsilon24 * (lj12 + lj12m6) * invDr2;

                            const double fX = fac * drX;
                            const double fY = fac * drY;
                            const double fZ = fac * drZ;

                            // TODO: consider newton3 if enabled
                            fxAcc += fX;
                            fyAcc += fY;
                            fzAcc += fZ;
                        }
                    }
                //}, fxAcc, fyAcc, fzAcc);
                }


                const double oldFx = soa1.template operator()<Particle_T::AttributeNames::forceX, true, false>(i);
                const double oldFy = soa1.template operator()<Particle_T::AttributeNames::forceY, true, false>(i);
                const double oldFz = soa1.template operator()<Particle_T::AttributeNames::forceZ, true, false>(i);

                const double newFx = oldFx + fxAcc;
                const double newFy = oldFy + fyAcc;
                const double newFz = oldFz + fzAcc;

                soa1.template operator()<Particle_T::AttributeNames::forceX, true, false>(i) = newFx;
                soa1.template operator()<Particle_T::AttributeNames::forceY, true, false>(i) = newFy;
                soa1.template operator()<Particle_T::AttributeNames::forceZ, true, false>(i) = newFz;
            }
        });
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