// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of cuDOLFINX
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cuda.h>
#include <memory>
#include <string>
#include <vector>

namespace dolfinx
{

namespace CUDA
{
class Module;
class Kernel;

/// This class is a wrapper around a CUDA device context
class Context
{
public:
  /// Create a CUDA device context
  Context();

  /// Destructor
  ~Context();

  /// Copy constructor
  /// @param[in] context The object to be copied
  Context(const Context& context) = delete;

  /// Move constructor
  /// @param[in] context The object to be moved
  Context(Context&& context) = delete;

  /// Assignment operator
  /// @param[in] context The object to assign from
  Context& operator=(const Context& context) = delete;

  /// Move assignment operator
  /// @param[in] context The object to assign from
  Context& operator=(Context&& context) = delete;

  /// Return underlying CUDA device
  const CUdevice& device() const;

  /// Return underlying CUDA context
  CUcontext& context();

private:
  CUdevice _device;
  CUcontext _context;
};

/// This class is a wrapper around a module, which is obtained by
/// compiling PTX assembly to CUDA device code.
class Module
{
public:
  /// Create an empty module
  Module();

  /// Create a module
  Module(
    const CUDA::Context& cuda_context,
    const std::string& ptx,
    CUjit_target target,
    int num_module_load_options,
    CUjit_option* module_load_options,
    void** module_load_option_values,
    bool verbose,
    bool debug);

  /// Destructor
  ~Module();

  /// Copy constructor
  /// @param[in] module The object to be copied
  Module(const Module& module) = delete;

  /// Move constructor
  /// @param[in] module The object to be moved
  Module(Module&& module);

  /// Assignment operator
  /// @param[in] module The object to assign from
  Module& operator=(const Module& module) = delete;

  /// Move assignment operator
  /// @param[in] module The object to assign from
  Module& operator=(Module&& module);

  /// Get a device-side function from a loaded module
  CUfunction get_device_function(
    const std::string& device_function_name) const;

  /// Get info log for a loaded module
  const char* info_log() const {
    return _info_log; }

  /// Get error log for a loaded module
  const char* error_log() const {
    return _error_log; }

private:
  /// Handle to the CUDA module
  CUmodule _module;

  /// Size of the buffer for informational log messages
  size_t _info_log_size;

  /// Informational log messages related to loading the module
  char* _info_log;

  /// Size of the buffer for error log messages
  size_t _error_log_size;

  /// Error log messages related to loading the module
  char* _error_log;
};

/// Use the NVIDIA CUDA Runtime Compilation (nvrtc) library to compile
/// device-side code for a given CUDA program.
std::string compile_cuda_cpp_to_ptx(
  const char* program_name,
  int num_program_headers,
  const char** program_headers,
  const char** program_include_names,
  int num_compile_options,
  const char** compile_options,
  const char* program_src,
  const char* cudasrcdir,
  bool verbose);

void safeMemAlloc(CUdeviceptr* dptr, size_t bytesize);
void safeMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
void safeMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
void safeDeviceGetAttribute(int * res, CUdevice_attribute attr, CUdevice dev);
void safeCtxSynchronize();
void safeStreamCreate(CUstream* streamptr, unsigned int flags);

template <typename T> void safeVectorCreate(CUdeviceptr* dptr, std::vector<T> arr) {
  size_t bytesize = sizeof(T) * arr.size();
  safeMemAlloc(dptr, bytesize);
  safeMemcpyHtoD(*dptr, (void *)arr.data(), bytesize);
}

CUjit_target get_cujit_target(const Context& cuda_context);

} // namespace CUDA


} // namespace dolfinx
