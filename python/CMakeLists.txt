cmake_minimum_required(VERSION 3.18)

project(cudolfinx_nanobind)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(
  Python
  COMPONENTS Interpreter Development
  REQUIRED
)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND
    ${Python3_EXECUTABLE} -c
    "import os, sys, basix; sys.stdout.write(os.path.dirname(basix.__file__))"
  OUTPUT_VARIABLE BASIX_PY_DIR
  RESULT_VARIABLE BASIX_PY_COMMAND_RESULT
  ERROR_VARIABLE BASIX_ERROR_OUT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(Basix REQUIRED CONFIG HINTS ${BASIX_PY_DIR})

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

# Create the binding library nanobind handles its own calls to
# target_link_libraries
nanobind_add_module(
  cpp
  NOMINSIZE
  cudolfinx/wrappers/cudolfinx.cpp
  cudolfinx/wrappers/fem.cpp
  cudolfinx/wrappers/petsc.cpp
)

# Add strict compiler flags include(CheckCXXCompilerFlag)
# check_cxx_compiler_flag("-Wall -Werror -pedantic" HAVE_PEDANTIC)

# if(HAVE_PEDANTIC) # target_compile_options(cpp PRIVATE
# -Wall;-Werror;-pedantic) endif()

# Add DOLFINx libraries
target_link_libraries(cpp PRIVATE dolfinx)
target_link_libraries(cpp PRIVATE cudolfinx)
target_link_libraries(cpp PRIVATE CUDA::cuda_driver CUDA::nvrtc CUDA::cupti)
target_include_directories(cpp SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Check for petsc4py
execute_process(
  COMMAND ${Python_EXECUTABLE} -c
          "import petsc4py; print(petsc4py.get_include())"
  OUTPUT_VARIABLE PETSC4PY_INCLUDE_DIR
  RESULT_VARIABLE PETSC4PY_COMMAND_RESULT
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT PETSC4PY_COMMAND_RESULT)
  message(STATUS "Found petsc4py include directory at ${PETSC4PY_INCLUDE_DIR}")
  target_include_directories(cpp PRIVATE ${PETSC4PY_INCLUDE_DIR})
else()
  message(FATAL_ERROR "petsc4py could not be found.")
endif()

# Check for mpi4py
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import mpi4py; print(mpi4py.get_include())"
  OUTPUT_VARIABLE MPI4PY_INCLUDE_DIR
  RESULT_VARIABLE MPI4PY_COMMAND_RESULT
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT MPI4PY_COMMAND_RESULT)
  message(STATUS "Found mpi4py include directory at ${MPI4PY_INCLUDE_DIR}")
  target_include_directories(cpp PRIVATE ${MPI4PY_INCLUDE_DIR})
else()
  message(FATAL_ERROR "mpi4py could not be found.")
endif()

set_target_properties(cpp PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS cpp DESTINATION cudolfinx)
