# ------------------------------------------------------------------------------
include(GNUInstallDirs)

# ------------------------------------------------------------------------------
# Declare the library (target)
add_library(cudolfinx)

# ------------------------------------------------------------------------------
# Add source files to the target
set(CUDOLFINX_DIRS
    common
    fem
    la
    mesh
)

# Add source to dolfinx target, and get sets of header files
foreach(DIR ${CUDOLFINX_DIRS})
  add_subdirectory(${DIR})
endforeach()

# Set target include location (for build and installed)
target_include_directories(
  cudolfinx
  PUBLIC
    $<INSTALL_INTERFACE:include>
    "$<BUILD_INTERFACE:${CUDOLFINX_SOURCE_DIR};${CUDOLFINX_SOURCE_DIR}/cudolfinx>"
)

# ------------------------------------------------------------------------------
# Configure the common/version.h file
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/common/version.h.in common/version.h @ONLY
)

# ------------------------------------------------------------------------------
# Set target properties
set_target_properties(
  cudolfinx
  PROPERTIES VERSION ${CUDOLFINX_VERSION}
  SOVERSION ${CUDOLFINX_VERSION_MAJOR}.${CUDOLFINX_VERSION_MINOR}
)

# Add git revision flag to the one affected file
#set_source_files_properties(
#  common/defines.cpp
#  PROPERTIES
#    COMPILE_DEFINITIONS
#    "UFCX_SIGNATURE=\"${UFCX_SIGNATURE}\";CUDOLFINX_GIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\""
#)

# ------------------------------------------------------------------------------
# Set compiler options and definitions

# Set 'Developer' build type flags
target_compile_options(
	cudolfinx PRIVATE $<$<CONFIG:Developer>:${CUDOLFINX_CXX_DEVELOPER_FLAGS}>
)

# Add version to definitions (public)
target_compile_definitions(cudolfinx PUBLIC CUDOLFINX_VERSION="${CUDOLFINX_VERSION}")

# ------------------------------------------------------------------------------
# Add include directories and libraries of required packages

# UFCx
if(TARGET ufcx::ufcx)
  target_link_libraries(cudolfinx PUBLIC ufcx::ufcx)
else()
  target_include_directories(cudolfinx SYSTEM PUBLIC ${UFCX_INCLUDE_DIRS})
endif()

# Basix
target_link_libraries(cudolfinx PUBLIC Basix::basix)

# Boost
target_link_libraries(cudolfinx PUBLIC Boost::headers)
target_link_libraries(cudolfinx PUBLIC Boost::timer)

# MPI
target_link_libraries(cudolfinx PUBLIC MPI::MPI_CXX)

# HDF5
target_link_libraries(cudolfinx PUBLIC hdf5::hdf5)

# CUDA Toolkit
target_link_libraries(cudolfinx PRIVATE CUDA::cuda_driver CUDA::nvrtc CUDA::cupti)
target_include_directories(cudolfinx SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Dolfinx
target_link_libraries(cudolfinx PUBLIC dolfinx)

# ------------------------------------------------------------------------------
# Optional packages

# PETSc
if(CUDOLFINX_ENABLE_PETSC AND PETSC_FOUND)
  target_link_libraries(cudolfinx PUBLIC PkgConfig::PETSC)
  target_compile_definitions(cudolfinx PUBLIC HAS_PETSC)
endif()


# ------------------------------------------------------------------------------
# Install cudolfinx library and header files
install(
  TARGETS cudolfinx
  EXPORT CUDOLFINXTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
)

# Generate CUDOLFINXTargets.cmake
install(EXPORT CUDOLFINXTargets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cudolfinx)

# Install the header files
install(
  FILES cudolfinx.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT Development
)

foreach(DIR ${CUDOLFINX_DIRS})
  install(
    FILES ${HEADERS_${DIR}}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cudolfinx/${DIR}
    COMPONENT Development
  )
endforeach()

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/common/version.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cudolfinx/common
  COMPONENT Development
)

# ------------------------------------------------------------------------------
# Generate CMake config files (CUDOLFINXConfig{,Version}.cmake)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_BINARY_DIR}/cudolfinx/CUDOLFINXConfigVersion.cmake
  VERSION ${CUDOLFINX_VERSION}
  COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
	${CUDOLFINX_SOURCE_DIR}/cmake/templates/CUDOLFINXConfig.cmake.in
	${CMAKE_BINARY_DIR}/cudolfinx/CUDOLFINXConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cudolfinx
)

# Install CMake helper files
install(
	FILES ${CMAKE_BINARY_DIR}/cudolfinx/CUDOLFINXConfig.cmake
	${CMAKE_BINARY_DIR}/cudolfinx/CUDOLFINXConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cudolfinx
  COMPONENT Development
)

# ------------------------------------------------------------------------------
# Generate pkg-config file and install it

# Define packages that should be required by pkg-config file
set(PKG_REQUIRES "")

# Get link libraries and includes
get_target_property(
  PKGCONFIG_CUDOLFINX_TARGET_LINK_LIBRARIES cudolfinx INTERFACE_LINK_LIBRARIES
)
get_target_property(
  PKGCONFIG_CUDOLFINX_INCLUDE_DIRECTORIES cudolfinx
  INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
)

# Add imported targets to lists for creating pkg-config file
set(PKGCONFIG_CUDOLFINX_LIBS)

foreach(_target ${PKGCONFIG_CUDOLFINX_TARGET_LINK_LIBRARIES})
  if("${_target}" MATCHES "^[^<>]+$") # Skip "$<foo...>", which we get with
                                      # static libs
    if("${_target}" MATCHES "^.*::.*$")
      # Get include paths
      get_target_property(_inc_dirs ${_target} INTERFACE_INCLUDE_DIRECTORIES)

      if(_inc_dirs)
        list(APPEND PKGCONFIG_CUDOLFINX_INCLUDE_DIRECTORIES ${_inc_dirs})
      endif()

      # Get libraries
      get_target_property(_libs ${_target} INTERFACE_LINK_LIBRARIES)

      if(_libs)
        list(APPEND PKGCONFIG_CUDOLFINX_LIBS ${_libs})
      endif()

    else()
      # 'regular' libs, i.e. not imported targets
      list(APPEND PKGCONFIG_CUDOLFINX_LIBS ${_target})
    endif()

    # Special handling for compiled Boost imported targets
    if(("${_target}" MATCHES "^.*Boost::.*$") AND NOT "${_target}" STREQUAL
                                                  "Boost::headers"
    )
      get_target_property(_libs ${_target} IMPORTED_LOCATION_RELEASE)

      if(_libs)
        list(APPEND PKGCONFIG_CUDOLFINX_LIBS ${_libs})
      endif()
    endif()
  endif()
endforeach()

# Join include lists and remove duplicates
list(REMOVE_DUPLICATES PKGCONFIG_CUDOLFINX_INCLUDE_DIRECTORIES)
list(REMOVE_DUPLICATES PKGCONFIG_CUDOLFINX_LIBS)

# Convert include dirs to -I<incdir> form
foreach(_inc_dir ${PKGCONFIG_CUDOLFINX_INCLUDE_DIRECTORIES})
  set(PKG_INCLUDES "-I${_inc_dir} ${PKG_INCLUDES}")
endforeach()

# Get cudolfinx definitions
get_target_property(
  PKG_CUDOLFINX_DEFINITIONS cudolfinx INTERFACE_COMPILE_DEFINITIONS
)
set(PKG_DEFINITIONS)

foreach(_def ${PKG_DOLFINX_DEFINITIONS})
  set(PKG_DEFINITIONS "${PKG_DEFINITIONS} -D${_def}")
endforeach()

# Get basix definitions (this is required to propagate Basix definition to the
# pkg-config file, in the future Basix should create its own basix.pc file, see
# https://github.com/FEniCS/basix/issues/204)
get_target_property(
  PKG_BASIX_DEFINITIONS Basix::basix INTERFACE_COMPILE_DEFINITIONS
)

foreach(_def ${PKG_BASIX_DEFINITIONS})
  set(PKG_DEFINITIONS "${PKG_DEFINITIONS} -D${_def}")
endforeach()

# Convert compiler flags and definitions into space separated strings
string(REPLACE ";" " " PKG_CXXFLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE ";" " " PKG_LINKFLAGS "${CMAKE_EXE_LINKER_FLAGS}")

# Convert libraries to -L<libdir> -l<lib> form
foreach(_lib ${PKGCONFIG_CUDOLFINX_LIBS})
  # Add -Wl,option directives
  if("${_lib}" MATCHES "-Wl,[^ ]*")
    set(PKG_LINKFLAGS "${_lib} ${PKG_LINKFLAGS}")
  else()
    get_filename_component(_path ${_lib} DIRECTORY)
    get_filename_component(_name ${_lib} NAME_WE)
    string(REPLACE "lib" "" _name "${_name}")

    # Add libraries that matches the form -L<libdir> -l<lib>
    if(NOT "${_path}" STREQUAL "")
      set(PKG_LINKFLAGS "-L${_path} -l${_name} ${PKG_LINKFLAGS}")
    endif()
  endif()
endforeach()

# Remove duplicated link flags
separate_arguments(PKG_LINKFLAGS)
list(REMOVE_DUPLICATES PKG_LINKFLAGS)
string(REPLACE ";" " " PKG_LINKFLAGS "${PKG_LINKFLAGS}")

# Add additional link flags
foreach(_linkflag ${CUDOLFINX_LINK_FLAGS})
  set(PKG_LINKFLAGS "${PKG_LINKFLAGS} ${_linkflag}")
endforeach()

# Boost include dir (used as pkg-config variable)
get_target_property(
  BOOST_INCLUDE_DIR Boost::headers INTERFACE_INCLUDE_DIRECTORIES
)

# Configure and install pkg-config file
configure_file(
  ${CUDOLFINX_SOURCE_DIR}/cmake/templates/cudolfinx.pc.in
  ${CMAKE_BINARY_DIR}/dolfinx.pc @ONLY
)
install(
  FILES ${CMAKE_BINARY_DIR}/cudolfinx.pc
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
  COMPONENT Development
)

# ------------------------------------------------------------------------------
