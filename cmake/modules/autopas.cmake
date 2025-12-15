message(STATUS "Adding AutoPas")

include(FetchContent)

set(autopasRepoPath https://github.com/AutoPas/AutoPas.git)

FetchContent_Declare(
        autopasfetch
        GIT_REPOSITORY ${autopasRepoPath}
        GIT_TAG feature/kokkos-direct-sum
)

FetchContent_MakeAvailable(autopasfetch)

target_compile_options(autopas PRIVATE -w)
get_target_property(propval autopas INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(autopas SYSTEM PUBLIC "${propval}")