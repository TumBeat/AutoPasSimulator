# AutoPasSimulator

This project is sort of a clone of md-flexible.
It instantiates the AutoPas library and calls the respective functions (computeInteractions, forEachKokkos, ...) in order to benchmark performance.

The difference lies in the fact that this is able to compile for Kokkos_ENABLE_CUDA=TRUE and md-flexible is not...