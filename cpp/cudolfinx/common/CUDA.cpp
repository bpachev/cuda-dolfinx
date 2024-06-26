// Copyright (C) 2020 James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CUDA.h"

#if defined(HAS_CUDA_TOOLKIT)
#include <cuda.h>
#include <nvrtc.h>
#endif

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <iomanip>
#include <iterator>
#include <memory>
#include <string>
#include <stdexcept>

using namespace dolfinx;

#if defined(HAS_CUDA_TOOLKIT)
//-----------------------------------------------------------------------------
CUDA::Context::Context()
{
  CUresult cuda_err;
  const char * cuda_err_description;

  // Create a CUDA device context
  cuda_err = cuDeviceGet(&_device, 0);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuDeviceGet() failed with " + std::string(cuda_err_description) + " "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }

  cuda_err = cuDevicePrimaryCtxRetain(&_context, _device);
  // TODO for some reason on Frontera the above isn't working.. . 
  // Figure out why
  //cuda_err = cuCtxCreate(&_context, 0, _device);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuDevicePrimaryCtxRetain() failed with " + std::string(cuda_err_description) + " "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
}
//-----------------------------------------------------------------------------
CUDA::Context::~Context()
{
    //cuCtxDestroy(_context);
    cuDevicePrimaryCtxRelease(_device);
}
//-----------------------------------------------------------------------------
const CUdevice& CUDA::Context::device() const
{
  return _device;
}
//-----------------------------------------------------------------------------
CUcontext& CUDA::Context::context()
{
  return _context;
}
//-----------------------------------------------------------------------------
CUDA::Module::Module()
  : _module(nullptr)
  , _info_log_size(0)
  , _info_log(nullptr)
  , _error_log_size(0)
  , _error_log(nullptr)
{
}
//-----------------------------------------------------------------------------
CUDA::Module::Module(
  const CUDA::Context& cuda_context,
  const std::string& ptx,
  CUjit_target target,
  int num_module_load_options,
  CUjit_option* module_load_options,
  void** module_load_option_values,
  bool verbose,
  bool debug)
  : _module(nullptr)
  , _info_log_size(10*1024)
  , _info_log(new char[_info_log_size])
  , _error_log_size(10*1024)
  , _error_log(new char[_error_log_size])
{
  memset(_info_log, 0, _info_log_size);
  memset(_error_log, 0, _error_log_size);

  CUjit_option default_options[] = {
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    CU_JIT_INFO_LOG_BUFFER,
    CU_JIT_ERROR_LOG_BUFFER,
    CU_JIT_LOG_VERBOSE,
    CU_JIT_GENERATE_DEBUG_INFO,
    CU_JIT_GENERATE_LINE_INFO,
    CU_JIT_OPTIMIZATION_LEVEL,
    CU_JIT_TARGET_FROM_CUCONTEXT,
  };
  void* default_option_values[] = {
    (void*) _info_log_size,
    (void*) _error_log_size,
    (void*) _info_log,
    (void*) _error_log,
    verbose ? (void*) 1 : (void*) 0,
    debug ? (void*) 1 : (void*) 0,
    debug ? (void*) 0 : (void*) 1,
    debug ? (void*) 0 : (void*) 4,
    (void*) target,
  };
  int num_default_options =
    sizeof(default_options) /
    sizeof(*default_options);

  if (num_module_load_options == 0) {
    num_module_load_options = num_default_options;
    module_load_options = default_options;
    module_load_option_values = default_option_values;
  } else {
    delete[] _error_log;
    delete[] _info_log;
    throw std::runtime_error(
      "CUDA::Module(): Extra options not supported "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }

  CUresult cuda_err = cuModuleLoadDataEx(
    &_module, ptx.c_str(),
    num_module_load_options,
    module_load_options,
    module_load_option_values);

  _info_log_size = (size_t) module_load_option_values[0];
  _error_log_size = (size_t) module_load_option_values[1];
  if (_info_log_size > 0) {
    std::cerr << "Info log (" << _info_log_size << " bytes):\n";
    std::copy(
      _info_log, _info_log + _info_log_size,
      std::ostream_iterator<char>(std::cerr));
    std::cerr << std::endl;
  }
  if (_error_log_size > 0) {
    std::cerr << "Error log (" << _error_log_size << " bytes):\n";
    std::copy(
      _error_log, _error_log + _error_log_size,
      std::ostream_iterator<char>(std::cerr));
    std::cerr << std::endl;
  }

  if (cuda_err != CUDA_SUCCESS) {
    const char* cuda_err_description;
    cuGetErrorString(cuda_err, &cuda_err_description);
    delete[] _error_log;
    delete[] _info_log;
    throw std::runtime_error(
      "cuModuleLoadDataEx() failed with " + std::string(cuda_err_description) +
      " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
}
//-----------------------------------------------------------------------------
CUDA::Module::~Module()
{
  if (_info_log)
    delete[] _info_log;
  if (_error_log)
    delete[] _error_log;

  if (_module) {
    CUresult cuda_err;
    cuda_err = cuModuleUnload(_module);
    if (cuda_err != CUDA_SUCCESS) {
      const char* cuda_err_description;
      cuGetErrorString(cuda_err, &cuda_err_description);
      std::cerr <<
        "cuModuleUnload() failed with " <<
        cuda_err_description << std::endl;
    }
  }
}
//-----------------------------------------------------------------------------
CUDA::Module::Module(CUDA::Module&& module)
  : _module(module._module)
  , _info_log_size(module._info_log_size)
  , _info_log(module._info_log)
  , _error_log_size(module._error_log_size)
  , _error_log(module._error_log)
{
  module._module = nullptr;
  module._info_log_size = 0;
  module._info_log = nullptr;
  module._error_log_size = 0;
  module._error_log = nullptr;
}
//-----------------------------------------------------------------------------
CUDA::Module& CUDA::Module::operator=(CUDA::Module&& module)
{
  _module = module._module;
  _info_log_size = module._info_log_size;
  _info_log = module._info_log;
  _error_log_size = module._error_log_size;
  _error_log = module._error_log;
  module._module = nullptr;
  module._info_log_size = 0;
  module._info_log = nullptr;
  module._error_log_size = 0;
  module._error_log = nullptr;
  return *this;
}
//-----------------------------------------------------------------------------
/// Get a device-side function from a loaded module
CUfunction CUDA::Module::get_device_function(
  const std::string& device_function_name) const
{
  CUfunction device_function;
  CUresult cuda_err = cuModuleGetFunction(
    &device_function, _module, device_function_name.c_str());
  if (cuda_err != CUDA_SUCCESS) {
    const char* cuda_err_description;
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuModuleGetFunction() failed with " +
      std::string(cuda_err_description) + ": " + device_function_name + " "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
  return device_function;
}
//-----------------------------------------------------------------------------
/// Print a CUDA C++ program annotated with line numbers
std::ostream& print_cuda_cpp(
  std::ostream& o,
  const char* program_src)
{
  // Count the number of lines
  std::string program_src_str = std::string(program_src);
  size_t num_lines = std::count(
    program_src_str.begin(), program_src_str.end(), '\n') + 1;
  int field_width = ceil(log10(num_lines));

  // Print each line annotated with a line number
  std::stringstream program_src_ss(program_src_str);
  std::string program_src_line;
  for (size_t i = 1;
       std::getline(program_src_ss, program_src_line);
       i++)
  {
    o << std::setw(field_width) << i << "| "
      << program_src_line << "\n";
  }
  return o;
}
//-----------------------------------------------------------------------------
static int mkdirp(const char* path, mode_t mode)
{
  int err;

  // Get file attributes for the given path
  struct stat fattrs;
  err = stat(path, &fattrs);
  if (!err) {
    if (S_ISDIR(fattrs.st_mode) != 0) {
      // File exists, and it is a directory
      return 0;
    } else {
      // File exists, but it is not a directory
      return ENOTDIR;
    }
  } else if (errno != ENOENT) {
    // File exists, but could not get file attributes
    return err;
  }

  // If the path contains a slash, then create the parent directory
  const char* pathsep = strrchr(path, '/');
  if (pathsep) {
    char* parentdir = strndup(path, pathsep - path);
    err = mkdirp(parentdir, mode);
    if (err) {
      free(parentdir);
      return err;
    }
    free(parentdir);
  }

  // The file does not exist, so create a directory
  err = mkdir(path, mode);
  if (err)
    return err;
  return 0;
}
//-----------------------------------------------------------------------------
/// Use the NVIDIA CUDA Runtime Compilation (nvrtc) library to compile
/// device-side code for a given CUDA program.
std::string CUDA::compile_cuda_cpp_to_ptx(
  const char* program_name,
  int num_program_headers,
  const char** program_headers,
  const char** program_include_names,
  int num_compile_options,
  const char** compile_options,
  const char* program_src,
  const char* cudasrcdir,
  bool verbose)
{
  // Write the CUDA C++ source code to a file, if cudasrcdir is specified
  if (cudasrcdir) {
    int err = mkdirp(cudasrcdir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (err) {
      throw std::runtime_error(
        std::string(cudasrcdir) + ": " + std::string(strerror(errno)) + " "
        "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    std::string path = std::string(cudasrcdir) +
      std::string("/") + std::string(program_name) + std::string(".cu");
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
      throw std::runtime_error(
        path + ": " + std::string(strerror(errno)) + " "
        "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    fputs(program_src, f);
    fclose(f);

    if (verbose) {
      std::cout << "CUDA C++ code for " << program_name
                << " written to " << path << std::endl;
    }
  }

  // Create a CUDA C++ program based on the given source
  nvrtcResult nvrtc_err;
  nvrtcProgram program;
  nvrtc_err = nvrtcCreateProgram(
    &program, program_src, program_name,
    num_program_headers, program_headers,
    program_include_names);
  if (nvrtc_err != NVRTC_SUCCESS) {
    throw std::runtime_error(
      "nvrtcCreateProgram() failed with " +
      std::string(nvrtcGetErrorString(nvrtc_err)) + " "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }

  // Compile the CUDA C++ program
  nvrtcResult nvrtc_compile_err = nvrtcCompileProgram(
    program, num_compile_options, compile_options);
  if (nvrtc_compile_err != NVRTC_SUCCESS) {
    // If the compiler failed, obtain the compiler log
    std::string program_log;
    size_t log_size;
    nvrtc_err = nvrtcGetProgramLogSize(program, &log_size);
    if (nvrtc_err != NVRTC_SUCCESS) {
      program_log = std::string(
        "nvrtcGetProgramLogSize() failed with " +
        std::string(nvrtcGetErrorString(nvrtc_err)) + " "
        "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    } else {
      program_log.resize(log_size);
      nvrtc_err = nvrtcGetProgramLog(
        program, const_cast<char*>(program_log.c_str()));
      if (nvrtc_err != NVRTC_SUCCESS) {
        program_log = std::string(
          "nvrtcGetProgramLog() failed with " +
          std::string(nvrtcGetErrorString(nvrtc_err))) + " "
          "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__);
      }
      if (log_size > 0)
        program_log.resize(log_size-1);
    }
    nvrtcDestroyProgram(&program);

    std::stringstream ss;
    ss << "nvrtcCompileProgram() failed with "
       << nvrtcGetErrorString(nvrtc_compile_err) << "\n"
       << "CUDA C++ source code:\n"
       << std::string(60, '-') << "\n";
    print_cuda_cpp(ss, program_src);
    ss << std::string(60, '-') << "\n"
       << "NVRTC compiler log:\n"
       << std::string(60, '-') << "\n"
       << program_log << "\n"
       << std::string(60, '-') << "\n";
    throw std::runtime_error(ss.str());
  }

#if defined(NVRTC_PRINT_SOURCE)
  std::cerr << "CUDA C++ source code for "
            << program_name << ":\n"
            << std::string(60, '-') << "\n";
  print_cuda_cpp(std::cerr, program_src);
  std::cerr << std::string(60, '-') << "\n";
#endif

  // Get the size of the compiled PTX assembly
  size_t ptx_size;
  nvrtc_err = nvrtcGetPTXSize(program, &ptx_size);
  if (nvrtc_err != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&program);
    throw std::runtime_error(
      "nvrtcGetPTXSize() failed with " +
      std::string(nvrtcGetErrorString(nvrtc_err)) + " "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }

  // Fetch the compiled PTX assembly
  std::string ptx(ptx_size, 0);
  nvrtc_err = nvrtcGetPTX(
    program, const_cast<char*>(ptx.c_str()));
  if (nvrtc_err != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&program);
    throw std::runtime_error(
      "nvrtcGetPTX() failed with " +
      std::string(nvrtcGetErrorString(nvrtc_err)) + " "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }

  // Clean up the CUDA C++ program
  nvrtc_err = nvrtcDestroyProgram(&program);
  if (nvrtc_err != NVRTC_SUCCESS) {
    throw std::runtime_error(
      "nvrtcDestroyProgram() failed with " +
      std::string(nvrtcGetErrorString(nvrtc_err)) + " "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }

  return ptx;
}
//-----------------------------------------------------------------------------

void CUDA::safeMemAlloc(CUdeviceptr* dptr, size_t bytesize)
{
  const char * cuda_err_description;
  CUresult cuda_err = cuMemAlloc(dptr, bytesize);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }  
}

void CUDA::safeMemcpyDtoH(void * dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
  const char * cuda_err_description;
  CUresult cuda_err = cuMemcpyDtoH(dstHost, srcDevice, ByteCount);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
        "cuMemcpyDtoH() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  } 
}

void CUDA::safeMemcpyHtoD(CUdeviceptr dstDevice, void* srcHost, size_t ByteCount)
{
  const char * cuda_err_description;
  CUresult cuda_err = cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
}


void CUDA::safeDeviceGetAttribute(int * res, CUdevice_attribute attrib, CUdevice dev)
{
  const char * cuda_err_description;
  CUresult cuda_err = cuDeviceGetAttribute(res, attrib, dev);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
     throw std::runtime_error(
           "cuDeviceGetAttribute failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
}

void CUDA::safeCtxSynchronize()
{
  const char * cuda_err_description;
  CUresult cuda_err = cuCtxSynchronize();
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuCtxSynchronize() failed with " + std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }
}

void CUDA::safeStreamCreate(CUstream* streamptr, unsigned int flags)
{
  const char * cuda_err_description;
  CUresult cuda_err = cuStreamCreate(streamptr, flags);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuCtxSynchronize() failed with " + std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }
}

CUjit_target CUDA::get_cujit_target(const CUDA::Context& cuda_context)
{
  int compute_major, compute_minor;
  CUDA::safeDeviceGetAttribute(
            &compute_major,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            cuda_context.device());
  CUDA::safeDeviceGetAttribute(
            &compute_minor,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            cuda_context.device());

  int compute_level = 10*compute_major + compute_minor;
  std::map<int, CUjit_target> targets = {
        {30, CU_TARGET_COMPUTE_30}, {32, CU_TARGET_COMPUTE_32},
        {35, CU_TARGET_COMPUTE_35}, {37, CU_TARGET_COMPUTE_37},
        {50, CU_TARGET_COMPUTE_50}, {52, CU_TARGET_COMPUTE_52},
        {53, CU_TARGET_COMPUTE_53}, {60, CU_TARGET_COMPUTE_60},
        {61, CU_TARGET_COMPUTE_61}, {62, CU_TARGET_COMPUTE_62},
        {70, CU_TARGET_COMPUTE_70}, {72, CU_TARGET_COMPUTE_72},
        {75, CU_TARGET_COMPUTE_75}, {80, CU_TARGET_COMPUTE_80},
        {86, CU_TARGET_COMPUTE_86}, {87, CU_TARGET_COMPUTE_87},
        {89, CU_TARGET_COMPUTE_89}, {90, CU_TARGET_COMPUTE_90}
  }; 
 
  auto it = targets.find(compute_level);
  if (it != targets.end()) {
      return it->second;
  }
  else {
     std::cout << "Unrecognized compute level " << compute_level << "." << std::endl;
     it = targets.lower_bound(compute_level);
     if (it == targets.begin()) throw std::runtime_error("Compute level is too low for fallback!");
     else {
       it--;
       std::cout << " Falling back to " << it->first << "." << std::endl;
       return it->second;
     }
  }
}

#endif
