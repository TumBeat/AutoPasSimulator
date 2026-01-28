/**
 * @file computeInteractionsFunctorKokkos.cpp
 * @date 11.12.2025
 * @author Luis Gall
 */

#include <autopas/AutoPasImpl.h>
#include <utils/KokkosParticle.h>
#include <utils/FunctorKokkos.h>

#ifdef KOKKOS_ENABLE_CUDA
template bool autopas::AutoPas<KokkosParticle>::computeInteractions(FunctorKokkos<KokkosParticle, Kokkos::CudaSpace> *);
#else
template bool autopas::AutoPas<KokkosParticle>::computeInteractions(FunctorKokkos<KokkosParticle, Kokkos::HostSpace> *);
#endif
