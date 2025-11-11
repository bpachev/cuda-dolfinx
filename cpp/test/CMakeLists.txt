cmake_minimum_required(VERSION 3.18)

project(interpolate)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-O2 -g -pg")

find_package(Basix REQUIRED)

set(MPI_HOME "/usr/local")
find_package(MPI REQUIRED)

if(Basix_FOUND)
  message(STATUS "Found Basix at ${Basix_DIR}")
endif()

find_package(DOLFINX REQUIRED CONFIG)
if(DOLFINX_FOUND)
  message(STATUS "Found DOLFINx at ${DOLFINX_DIR}")
endif()

find_package(CUDOLFINX REQUIRED CONFIG)
if(CUDOLFINX_FOUND)
  message(STATUS "Found CUDOLFINx at ${CUDOLFINX_DIR}")
endif()

find_package(CUDAToolkit REQUIRED)


add_executable(interpolate_same_map interpolate_same_map.cpp)

function(link_interpolate name)
  target_link_libraries(${name} PRIVATE MPI::MPI_CXX)
  target_link_libraries(${name} PRIVATE dolfinx)
  target_link_libraries(${name} PRIVATE cudolfinx)
  target_link_libraries(${name} PRIVATE CUDA::cuda_driver CUDA::nvrtc CUDA::cupti)
endfunction()

link_interpolate(interpolate_same_map)


