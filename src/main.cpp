#include <iostream>

#include <autopas/AutoPasDecl.h>


#include <utils/ParticleType.h>
#include <utils/FunctorKokkos.h>
#include <utils/Setup.h>

extern template class autopas::AutoPas<ParticleType>;

const size_t N = 100;
const size_t ITERATIONS = 10;

template <class ReturnType, class FunctionType>
ReturnType applyWithChosenFunctor(FunctionType f) {
    return f(FunctorKokkos<ParticleType, Kokkos::CudaSpace>{3.});
}

int main(int argc, char** argv) {

    autopas::AutoPas_MPI_Init(&argc, &argv);
    autopas::AutoPas_Kokkos_Init(argc, argv);
    {
        autopas::AutoPas<ParticleType> autoPasInstance = autopas::AutoPas<ParticleType>(std::cout);

        utils::Setup::provideOptions(autoPasInstance);
        autoPasInstance.init();
        autoPasInstance.reserve(N);
        utils::Setup::fillParticles(autoPasInstance, N);

        // TODO: forEachKokkos() seems to produce some kind of issue with the dual view
        // autoPasInstance.forEachKokkos(KOKKOS_LAMBDA(int i, autopas::utils::KokkosStorage<ParticleType>& storage) {
        //     storage.operator()<ParticleType::AttributeNames::forceX, true, false>(i) = 12.5;
        // });

        for (int i = 0; i < ITERATIONS; ++i) {
            applyWithChosenFunctor<bool>([&](auto && functor) { return autoPasInstance.computeInteractions(&functor); });
        }

        autoPasInstance.forEachKokkos(KOKKOS_LAMBDA(int i, const autopas::utils::KokkosStorage<ParticleType>& storage) {
            storage.operator()<ParticleType::AttributeNames::forceX, true, false>(i) *= 0.5;
        });

        bool test = true;
        autoPasInstance.reduceKokkos<bool, Kokkos::LAnd<bool>>(KOKKOS_LAMBDA(int i, const autopas::utils::KokkosStorage<ParticleType>& storage, bool& local) {
            double fX = storage.operator()<ParticleType::AttributeNames::forceX, true, false>(i);
            if (fX == 10.) {
                local &= true;
            }
            else {
                local &= false;
            }
        }, test);

        for (auto& p : autoPasInstance) {
            double fX = p.operator()<ParticleType::AttributeNames::forceX>();
            std::cout << fX << std::endl;
        }

        autoPasInstance.finalize();
    }
    autopas::AutoPas_MPI_Finalize();
    autopas::AutoPas_Kokkos_Finalize();

    return 0;
}