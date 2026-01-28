#include <iostream>

#include <autopas/AutoPasDecl.h>


#include <utils/KokkosParticle.h>
#include <utils/FunctorKokkos.h>
#include <utils/Setup.h>
#include "utils/Configuration.h"

extern template class autopas::AutoPas<KokkosParticle>;

#ifdef KOKKOS_ENABLE_CUDA
using DeviceSpace = Kokkos::CudaSpace;
using ForEachSpace = Kokkos::CudaSpace;
constexpr bool forEachHostFlag = false;
#else
using DeviceSpace = Kokkos::HostSpace;
using ForEachSpace = Kokkos::HostSpace;
constexpr bool forEachHostFlag = true;
#endif

template <class ReturnType, class FunctionType>
ReturnType applyWithChosenFunctor(FunctionType f) {
#ifdef KOKKOS_ENABLE_CUDA
    return f(FunctorKokkos<KokkosParticle, DeviceSpace>{3.});
#else
    return f(FunctorKokkos<KokkosParticle, DeviceSpace>{3.});
#endif
}

int main(int argc, char** argv) {

    autopas::AutoPas_MPI_Init(&argc, &argv);
    autopas::AutoPas_Kokkos_Init(argc, argv);
    {
        Configuration config {};
        config.parseConfig(argc, argv);
        autopas::AutoPas<KokkosParticle> autoPasInstance = autopas::AutoPas<KokkosParticle>(std::cout);

        std::cout << typeid(KokkosParticle::ParticleSoAFloatPrecision).name() << std::endl;

        // TODO: options for disabling tuning completely
        utils::Setup::provideOptions(autoPasInstance, config);
        autoPasInstance.init();
        autoPasInstance.reserve(config.getNumParticles(), config.getNumHalos());
        utils::Setup::fillParticles(autoPasInstance, config);

        auto positionTimer = autopas::utils::Timer();
        auto interactionsTimer = autopas::utils::Timer();
        auto velocityTimer = autopas::utils::Timer();

        double deltaT = config.getDeltaT();
        size_t iterations = config.getNumIterations();

        for (int i = 0; i < iterations; i++) {
            // 1. Position Update and Force reset
            positionTimer.start();
            Kokkos::Profiling::pushRegion("Position Update");
            autoPasInstance.forEachKokkos<ForEachSpace::execution_space>(KOKKOS_LAMBDA(int i, const autopas::utils::KokkosStorage<KokkosParticle>& storage) {

                const typename KokkosParticle::ParticleSoAFloatPrecision mass = storage.operator()<KokkosParticle::AttributeNames::mass, true, forEachHostFlag>(i);
                typename KokkosParticle::ParticleSoAFloatPrecision vX = storage.operator()<KokkosParticle::AttributeNames::velocityX, true, forEachHostFlag>(i);
                typename KokkosParticle::ParticleSoAFloatPrecision vY = storage.operator()<KokkosParticle::AttributeNames::velocityY, true, forEachHostFlag>(i);
                typename KokkosParticle::ParticleSoAFloatPrecision vZ = storage.operator()<KokkosParticle::AttributeNames::velocityZ, true, forEachHostFlag>(i);

                typename KokkosParticle::ParticleSoAFloatPrecision fX = storage.operator()<KokkosParticle::AttributeNames::forceX, true, forEachHostFlag>(i);
                typename KokkosParticle::ParticleSoAFloatPrecision fY = storage.operator()<KokkosParticle::AttributeNames::forceY, true, forEachHostFlag>(i);
                typename KokkosParticle::ParticleSoAFloatPrecision fZ = storage.operator()<KokkosParticle::AttributeNames::forceZ, true, forEachHostFlag>(i);

                storage.operator()<KokkosParticle::AttributeNames::oldForceX, true, forEachHostFlag>(i) = fX;
                storage.operator()<KokkosParticle::AttributeNames::oldForceY, true, forEachHostFlag>(i) = fY;
                storage.operator()<KokkosParticle::AttributeNames::oldForceZ, true, forEachHostFlag>(i) = fZ;

                // No global force, therefore 0
                storage.operator()<KokkosParticle::AttributeNames::forceX, true, forEachHostFlag>(i) = 5.;
                storage.operator()<KokkosParticle::AttributeNames::forceY, true, forEachHostFlag>(i) = 5.;
                storage.operator()<KokkosParticle::AttributeNames::forceZ, true, forEachHostFlag>(i) = 5.;

                vX *= deltaT;
                vY *= deltaT;
                vZ *= deltaT;

                fX *= (deltaT * deltaT / (2 * mass));
                fY *= (deltaT * deltaT / (2 * mass));
                fZ *= (deltaT * deltaT / (2 * mass));

                const typename KokkosParticle::ParticleSoAFloatPrecision displacementX = vX + fX;
                const typename KokkosParticle::ParticleSoAFloatPrecision displacementY = vY + fY;
                const typename KokkosParticle::ParticleSoAFloatPrecision displacementZ = vZ + fZ;

                storage.operator()<KokkosParticle::AttributeNames::posX, true, forEachHostFlag>(i) = displacementX + storage.operator()<KokkosParticle::AttributeNames::posX, true, forEachHostFlag>(i);
                storage.operator()<KokkosParticle::AttributeNames::posY, true, forEachHostFlag>(i) = displacementY + storage.operator()<KokkosParticle::AttributeNames::posY, true, forEachHostFlag>(i);
                storage.operator()<KokkosParticle::AttributeNames::posZ, true, forEachHostFlag>(i) = displacementZ + storage.operator()<KokkosParticle::AttributeNames::posZ, true, forEachHostFlag>(i);

            }, autopas::IteratorBehavior::owned);
            Kokkos::Profiling::popRegion();
            positionTimer.stop();

            // 2. Compute particle interactions based on the defined functor
            interactionsTimer.start();
            Kokkos::Profiling::pushRegion("Force Kernel");
            applyWithChosenFunctor<bool>([&](auto && functor) { return autoPasInstance.computeInteractions(&functor); });
            interactionsTimer.stop();

            /*
            bool test = false;
            autoPasInstance.reduceKokkos<ForEachSpace::execution_space, bool, Kokkos::LOr<bool>>(KOKKOS_LAMBDA(int i, const autopas::utils::KokkosStorage<ParticleType>& storage, bool& local) {
                const auto fX = storage.operator()<ParticleType::AttributeNames::forceX, true, forEachHost>(i);
                local |=  (fX != 5.);
            }, test, autopas::IteratorBehavior::owned);
            */

            // 3. Velocity update
            velocityTimer.start();
            Kokkos::Profiling::pushRegion("Velocity Update");
            autoPasInstance.forEachKokkos<ForEachSpace::execution_space>(KOKKOS_LAMBDA(int i, const autopas::utils::KokkosStorage<KokkosParticle>& storage) {

                const typename KokkosParticle::ParticleSoAFloatPrecision mass = storage.operator()<KokkosParticle::AttributeNames::mass, true, forEachHostFlag>(i);

                const typename KokkosParticle::ParticleSoAFloatPrecision fX = storage.operator()<KokkosParticle::AttributeNames::forceX, true, forEachHostFlag>(i);
                const typename KokkosParticle::ParticleSoAFloatPrecision fY = storage.operator()<KokkosParticle::AttributeNames::forceY, true, forEachHostFlag>(i);
                const typename KokkosParticle::ParticleSoAFloatPrecision fZ = storage.operator()<KokkosParticle::AttributeNames::forceZ, true, forEachHostFlag>(i);

                const typename KokkosParticle::ParticleSoAFloatPrecision oldFx = storage.operator()<KokkosParticle::AttributeNames::oldForceX, true, forEachHostFlag>(i);
                const typename KokkosParticle::ParticleSoAFloatPrecision oldFy = storage.operator()<KokkosParticle::AttributeNames::oldForceY, true, forEachHostFlag>(i);
                const typename KokkosParticle::ParticleSoAFloatPrecision oldFz = storage.operator()<KokkosParticle::AttributeNames::oldForceZ, true, forEachHostFlag>(i);

                const typename KokkosParticle::ParticleSoAFloatPrecision vUpdateX = (fX + oldFx) * (deltaT / (2 * mass));
                const typename KokkosParticle::ParticleSoAFloatPrecision vUpdateY = (fY + oldFy) * (deltaT / (2 * mass));
                const typename KokkosParticle::ParticleSoAFloatPrecision vUpdateZ = (fZ + oldFz) * (deltaT / (2 * mass));

                storage.operator()<KokkosParticle::AttributeNames::velocityX, true, forEachHostFlag>(i) = vUpdateX + storage.operator()<KokkosParticle::AttributeNames::velocityX, true, forEachHostFlag>(i);
                storage.operator()<KokkosParticle::AttributeNames::velocityY, true, forEachHostFlag>(i) = vUpdateY + storage.operator()<KokkosParticle::AttributeNames::velocityX, true, forEachHostFlag>(i);
                storage.operator()<KokkosParticle::AttributeNames::velocityZ, true, forEachHostFlag>(i) = vUpdateZ + storage.operator()<KokkosParticle::AttributeNames::velocityX, true, forEachHostFlag>(i);

            }, autopas::IteratorBehavior::owned);
            Kokkos::Profiling::popRegion();
            velocityTimer.stop();


            /*
            for (auto p = autoPasInstance.begin(autopas::IteratorBehavior::owned); p.isValid(); ++p) {
                double fX = p->operator()<ParticleType::AttributeNames::forceX>();
                std::cout << fX << std::endl;
            }
            */

        }

        std::cout << "1. Update: " << positionTimer.getTotalTime() << std::endl;
        std::cout << "2. Update: " << interactionsTimer.getTotalTime() << std::endl;
        std::cout << "3. Update: " << velocityTimer.getTotalTime() << std::endl;

        autoPasInstance.finalize();
    }
    autopas::AutoPas_MPI_Finalize();
    autopas::AutoPas_Kokkos_Finalize();

    return 0;
}