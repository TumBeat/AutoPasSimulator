#include <iostream>

#include <autopas/AutoPasDecl.h>


#include <utils/ParticleType.h>
#include <utils/FunctorKokkos.h>
#include <utils/Setup.h>

extern template class autopas::AutoPas<ParticleType>;

const size_t N = 100;

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

        applyWithChosenFunctor<bool>([&](auto && functor) { return autoPasInstance.computeInteractions(&functor); });

        autoPasInstance.finalize();
    }
    autopas::AutoPas_MPI_Finalize();
    autopas::AutoPas_Kokkos_Finalize();

    return 0;
}