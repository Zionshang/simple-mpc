cmake_minimum_required(VERSION 3.10.0)
project(simple_mpc VERSION 0.1.0 LANGUAGES CXX)

list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_PYTHON_INTERFACE "Build Python bindings" ON)
option(BUILD_BENCHMARK "Build benchmarks" OFF)
option(BUILD_TESTING "Build tests" OFF)

find_package(OpenMP REQUIRED)
find_package(pinocchio REQUIRED)
find_package(aligator 0.16 REQUIRED)
find_package(proxsuite REQUIRED)
find_package(tsid 1.9.0 REQUIRED)
find_package(ndcurves REQUIRED CONFIG)

file(GLOB_RECURSE SIMPLE_MPC_SRC CONFIGURE_DEPENDS src/*.cpp)
add_library(simple-mpc SHARED ${SIMPLE_MPC_SRC})
target_include_directories(simple-mpc PUBLIC include)
target_link_libraries(simple-mpc
  PUBLIC
  OpenMP::OpenMP_CXX
  pinocchio::pinocchio
  aligator::aligator
  ndcurves::ndcurves
  tsid::tsid
)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

if(BUILD_BENCHMARK)
  add_subdirectory(benchmark)
endif()

if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(bindings)
endif()
