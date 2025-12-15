/**
 * @file computeInteractionsFunctorKokkos.cpp
 * @date 11.12.2025
 * @author Luis Gall
 */

#include <autopas/AutoPasImpl.h>
#include <utils/ParticleType.h>
#include <utils/FunctorKokkos.h>

template bool autopas::AutoPas<ParticleType>::computeInteractions(FunctorKokkos<ParticleType, Kokkos::CudaSpace> *);
