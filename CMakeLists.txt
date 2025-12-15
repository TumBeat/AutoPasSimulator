cmake_minimum_required(VERSION 3.14)
project(AutoPasSimulator)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
include(autopas)

file(GLOB_RECURSE
    SRC
        "src/*.cpp"
        "src/*.h"
)

add_executable(AutoPasSimulator ${SRC})

target_include_directories(AutoPasSimulator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(AutoPasSimulator
        PUBLIC
        autopas
        autopasTools
)