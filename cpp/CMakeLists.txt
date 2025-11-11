# ------------------------------------------------------------------------------
# Top level CMakeLists.txt file for DOLFINx
cmake_minimum_required(VERSION 3.19)

# ------------------------------------------------------------------------------
# Set project name and version number
project(CUDOLFINX VERSION "0.10.0")

set(DOXYGEN_CUDOLFINX_VERSION
    ${CUDOLFINX_VERSION}
    CACHE STRING "Version for Doxygen" FORCE
)

# ------------------------------------------------------------------------------
# Use C++20
set(CMAKE_CXX_STANDARD 20)

# Require C++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Do not enable compler-specific extensions
set(CMAKE_CXX_EXTENSIONS OFF)

enable_language(CUDA)
set(CMAKE_CUDA_FLAGS " -lineinfo")
# ------------------------------------------------------------------------------
# Get GIT changeset, if available
find_program(GIT_FOUND git)

if(GIT_FOUND)
  # Get the commit hash of the working branch
  execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
else()
  set(GIT_COMMIT_HASH "unknown")
endif()

# ------------------------------------------------------------------------------
# General configuration

# Set location of our FindFoo.cmake modules
set(CMAKE_MODULE_PATH "${CUDOLFINX_SOURCE_DIR}/cmake/modules")

# Make sure CMake uses the correct DOLFINConfig.cmake for tests and demos
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_BINARY_DIR}/cudolfinx)

# ------------------------------------------------------------------------------
# Configurable options for how we want to build
include(FeatureSummary)

option(BUILD_SHARED_LIBS "Build CUDOLFINx with shared libraries." ON)
add_feature_info(
 BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build CUDOLFINx with shared libraries."
)

option(CUDOLFINX_SKIP_BUILD_TESTS
       "Skip build tests for testing usability of dependency packages." OFF
)
add_feature_info(
  CUDOLFINX_SKIP_BUILD_TESTS CUDOLFINX_SKIP_BUILD_TESTS
  "Skip build tests for testing usability of dependency packages."
)

# Add shared library paths so shared libs in non-system paths are found
option(CMAKE_INSTALL_RPATH_USE_LINK_PATH
       "Add paths to linker search and installed rpath." ON
)
add_feature_info(
  CMAKE_INSTALL_RPATH_USE_LINK_PATH CMAKE_INSTALL_RPATH_USE_LINK_PATH
  "Add paths to linker search and installed rpath."
)

# Control UFCx discovery
option(
  CUDOLFINX_UFCX_PYTHON
  "Enable UFCx discovery using Python. Disable if UFCx should be found using CMake."
  ON
)
add_feature_info(
  CUDOLFINX_UFCX_PYTHON
  CUDOLFINX_UFCX_PYTHON
  "Enable UFCx discovery using Python. Disable if UFCx should be found using a CMake config file."
)

# ------------------------------------------------------------------------------
# Enable or disable optional packages


if(CUDOLFINX_ENABLE_PETSC)
  set(_REQUIRE_PETSC
      TRUE
      CACHE BOOL "Is PETSc REQUIRED?"
  )
else()
  set(_REQUIRE_PETSC
      FALSE
      CACHE BOOL "Is PETSc REQUIRED?"
  )
endif()

option(CUDOLFINX_ENABLE_PETSC "Compile with support for PETSc." ON)
set_package_properties(
  PETSc PROPERTIES
  TYPE RECOMMENDED
  DESCRIPTION "Portable, Extensible Toolkit for Scientific Computation"
  URL "https://petsc.org/"
  PURPOSE "Linear and nonlinear solvers"
)


# ------------------------------------------------------------------------------
# Check for MPI
find_package(MPI 3 REQUIRED)

# ------------------------------------------------------------------------------
# Compiler flags

# Default build type (can be overridden by user)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "RelWithDebInfo"
      CACHE
        STRING
        "Choose the type of build, options are: Debug Developer MinSizeRel Release RelWithDebInfo."
        FORCE
  )
endif()

# Check for some compiler flags
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-pipe HAVE_PIPE)

if(HAVE_PIPE)
  list(APPEND CUDOLFINX_CXX_DEVELOPER_FLAGS -pipe)
endif()

# Add some strict compiler checks
check_cxx_compiler_flag("-Wall -Werror -Wextra -pedantic" HAVE_PEDANTIC)

if(HAVE_PEDANTIC)
  list(APPEND CUDOLFINX_CXX_DEVELOPER_FLAGS -Wall;-Werror;-Wextra;-pedantic)
endif()

# Debug flags
check_cxx_compiler_flag(-g HAVE_DEBUG)

if(HAVE_DEBUG)
  list(APPEND CUDOLFINX_CXX_DEVELOPER_FLAGS -g)
endif()

# Optimisation
check_cxx_compiler_flag(-O2 HAVE_O2_OPTIMISATION)

if(HAVE_O2_OPTIMISATION)
  list(APPEND CUDOLFINX_CXX_DEVELOPER_FLAGS -O2)
endif()

# ------------------------------------------------------------------------------
# Find required packages

# Use Python for detecting UFCx and Basix
find_package(
  Python3
  COMPONENTS Interpreter
  QUIET
)

# Check for Basix Note: Basix may be installed as a standalone C++ library, or
# in the Basix Python module tree
if(Python3_Interpreter_FOUND)
  message(STATUS "Checking for basix hints with ${Python3_EXECUTABLE}")
  execute_process(
    COMMAND
      ${Python3_EXECUTABLE} -c
      "import basix, os, sys; sys.stdout.write(os.path.dirname(basix.__file__))"
    OUTPUT_VARIABLE BASIX_PY_DIR
    RESULT_VARIABLE BASIX_PY_COMMAND_RESULT
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(BASIX_PY_DIR)
    message(STATUS "Adding ${BASIX_PY_DIR} to Basix search hints")

    # Basix installed from manylinux wheel
    if(IS_DIRECTORY ${BASIX_PY_DIR}/../fenics_basix.libs)
      set(CMAKE_INSTALL_RPATH ${BASIX_PY_DIR}/../fenics_basix.libs)
    endif()
  endif()
endif()

find_package(Basix 0.10 REQUIRED CONFIG HINTS ${BASIX_PY_DIR})
set_package_properties(
  basix PROPERTIES
  TYPE REQUIRED
  DESCRIPTION "FEniCS tabulation library"
  URL "https://github.com/fenics/basix"
)

find_package(DOLFINX 0.10 REQUIRED CONFIG)
set_package_properties(
  DOLFINX PROPERTIES
  TYPE REQUIRED
  DESCRIPTION "Dynamic Object-oriented Library for FINite element computation"
  URL "https://github.com/fenics/basix"
)

# Check for HDF5
set(HDF5_PREFER_PARALLEL TRUE)
set(HDF5_FIND_DEBUG TRUE)
find_package(HDF5 REQUIRED COMPONENTS C)

if(NOT HDF5_IS_PARALLEL)
  message(
    FATAL_ERROR
      "Found serial HDF5 build, MPI HDF5 build required, try setting HDF5_DIR or HDF5_ROOT"
  )
endif()

set_package_properties(
  HDF5 PROPERTIES
  TYPE REQUIRED
  DESCRIPTION "Hierarchical Data Format 5 (HDF5)"
  URL "https://www.hdfgroup.org/HDF5"
)

# Check for UFC Note: we use the case (ufcx vs UFCx) elsewhere to determine by
# which method UFCx was found
if(NOT CUDOLFINX_UFCX_PYTHON)
  # Check in CONFIG mode, i.e. look for installed ufcxConfig.cmake
  find_package(ufcx 0.10 REQUIRED CONFIG)
else()
  # Check in MODULE mode (using FindUFCX.cmake)
  find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED
  )
  find_package(UFCx 0.10 REQUIRED MODULE)
endif()

set_package_properties(
  UFCx PROPERTIES
  TYPE REQUIRED
  DESCRIPTION "Interface for form-compilers (part of FFCx)"
  URL "https://github.com/fenics/ffcx"
)

find_package(CUDAToolkit REQUIRED)

set_package_properties(CUDAToolkit PROPERTIES TYPE OPTIONAL
  DESCRIPTION "Parallel computing platform for GPUs"
  URL "https://developer.nvidia.com/cuda-toolkit"
  PURPOSE "Enables GPU-accelerated computing"
)


# ------------------------------------------------------------------------------
# Find optional packages

if(CUDOLFINX_ENABLE_PETSC)
  find_package(PkgConfig REQUIRED)
  set(ENV{PKG_CONFIG_PATH}
      "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig:$ENV{PETSC_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}"
  )
  if(_REQUIRE_PETSC)
    pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc>=3.15 petsc>=3.15)
  else()
    pkg_search_module(PETSC OPTIONAL IMPORTED_TARGET PETSc>=3.15 petsc>=3.15)
  endif()

  # Setting for FeatureSummary
  if(PETSC_FOUND)
    message(
      STATUS "Found PETSc version ${PETSC_VERSION}, prefix: ${PETSC_PREFIX}"
    )
    set_property(GLOBAL APPEND PROPERTY PACKAGES_FOUND PETSc)
  else()
    set_property(GLOBAL APPEND PROPERTY PACKAGES_NOT_FOUND PETSc)
  endif()
endif()

# ------------------------------------------------------------------------------
# Print summary of found and not found optional packages
feature_summary(WHAT ALL)



# ------------------------------------------------------------------------------
# Installation of DOLFINx library
add_subdirectory(cudolfinx)

# ------------------------------------------------------------------------------
# Generate and install helper file cudolfinx.conf

# FIXME: Can CMake provide the library path name variable?
if(APPLE)
  set(OS_LIBRARY_PATH_NAME "DYLD_LIBRARY_PATH")
else()
  set(OS_LIBRARY_PATH_NAME "LD_LIBRARY_PATH")
endif()

# FIXME: not cross-platform compatible Create and install cudolfinx.conf file
configure_file(
  ${CUDOLFINX_SOURCE_DIR}/cmake/templates/cudolfinx.conf.in
  ${CMAKE_BINARY_DIR}/cudolfinx.conf @ONLY
)
install(
  FILES ${CMAKE_BINARY_DIR}/cudolfinx.conf
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cudolfinx
  COMPONENT Development
)

# ------------------------------------------------------------------------------
# Add "make uninstall" target
configure_file(
  "${CUDOLFINX_SOURCE_DIR}/cmake/templates/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake" IMMEDIATE @ONLY
)

add_custom_target(
  uninstall "${CMAKE_COMMAND}" -P
            "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
)

# ------------------------------------------------------------------------------
# Print post-install message
add_subdirectory(cmake/post-install)

# ------------------------------------------------------------------------------
