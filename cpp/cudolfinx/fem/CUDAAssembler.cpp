// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of cuDOLFINX
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <cudolfinx/common/CUDA.h>
#include <cudolfinx/fem/CUDAAssembler.h>
#include <cudolfinx/fem/CUDADirichletBC.h>
#include <cudolfinx/fem/CUDADofMap.h>
#include <cudolfinx/fem/CUDAFormIntegral.h>
#include <cudolfinx/fem/CUDAFormConstants.h>
#include <cudolfinx/fem/CUDAFormCoefficients.h>
#include <cudolfinx/la/CUDAMatrix.h>
#include <cudolfinx/la/CUDASeqMatrix.h>
#include <cudolfinx/la/CUDAVector.h>
#include <cudolfinx/mesh/CUDAMesh.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/Function.h>

#include <cuda.h>
#include <functional>
#include <memory>
#include <petscmat.h>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::fem;

namespace
{

/// CUDA C++ code for setting matrix entries to zero
std::string cuda_kernel_zero_double_array(void)
{
  return
    "/**\n"
    " * `zero_double_array()` sets each entry in an array of doubles to zero.\n"
    " */\n"
    "extern \"C\" __global__ void zero_double_array(\n"
    "  int num_entries,\n"
    "  double* values)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "  for (size_t i = thread_idx;\n"
    "    i < num_entries;\n"
    "    i += blockDim.x * gridDim.x)\n"
    "  {\n"
    "    values[i] = 0.0;\n"
    "  }\n"
    "}\n";
}

/// CUDA C++ code for packing form coefficients
std::string cuda_kernel_pack_coefficients(void)
{
  return
    "/**\n"
    " * `pack_coefficients()` packs coefficient values for a form.\n"
    " */\n"
    "extern \"C\" __global__ void pack_coefficients(\n"
    "  int num_coefficients,\n"
    "  const int* dofmaps_num_dofs_per_cell,\n"
    "  const int** dofmaps_dofs_per_cell,\n"
    "  const int* coefficient_values_offsets,\n"
    "  const double** coefficient_values,\n"
    "  const int* coefficient_indices,\n"
    "  int num_indices,\n"
    "  int num_cells,\n"
    "  int num_packed_coefficient_values_per_cell,\n"
    "  double* packed_values)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "  for (int index = 0; index < num_indices; index++) {\n"
    "    int j = coefficient_indices[index];\n"
    "    int num_dofs_per_cell = dofmaps_num_dofs_per_cell[j];\n"
    "    const int* dofs_per_cell = dofmaps_dofs_per_cell[j];\n"
    "    int offset = coefficient_values_offsets[j];\n"
    "    const double* values = coefficient_values[j];\n"
    "    for (size_t l = thread_idx;\n"
    "      l < num_cells*num_dofs_per_cell;\n"
    "      l += blockDim.x * gridDim.x)\n"
    "    {\n"
    "      int i = l / num_dofs_per_cell;\n"
    "      int k = l % num_dofs_per_cell;\n"
    "      int dof = dofs_per_cell[i*num_dofs_per_cell+k];\n"
    "      packed_values[i*num_packed_coefficient_values_per_cell +\n"
    "        offset + k] = values[dof];\n"
    "    }\n"
    "  }\n"
    "}\n";
}

/// CUDA C++ code for imposing Dirichlet boundary conditions on vectors
std::string cuda_kernel_set_bc(void)
{
  return
    "/**\n"
    " * `set_bc()` imposes Dirichlet boundary conditions on a vector.\n"
    " */\n"
    "extern \"C\" __global__ void set_bc(\n"
    "  int num_boundary_dofs,\n"
    "  const int* __restrict__ boundary_dofs,\n"
    "  const int* __restrict__ boundary_value_dofs,\n"
    "  const double* __restrict__ g,\n"
    "  const double* __restrict__ x0,\n"
    "  double scale,\n"
    "  int num_values,\n"
    "  double* __restrict__ values)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "  for (size_t i = thread_idx;\n"
    "    i < num_boundary_dofs;\n"
    "    i += blockDim.x * gridDim.x)\n"
    "  {\n"
    "    if (boundary_dofs[i] >= num_values) continue;\n"
    "    double _x0 = (x0) ? x0[boundary_dofs[i]] : 0.0;\n"
    "    values[boundary_dofs[i]] =\n"
    "      scale * (g[boundary_value_dofs[i]] - _x0);\n"
    "  }\n"
    "}\n";
}

/// CUDA C++ code for setting values to diagonal matrix entries
/// TODO support both adding and setting
/// The usual use case is setting
std::string cuda_kernel_set_diagonal(void)
{
  return
    "/**\n"
    " * `set_diagonal()` sets a value for given diagonal matrix entries.\n"
    " */\n"
    "extern \"C\" __global__ void set_diagonal(\n"
    "  int num_diagonals,\n"
    "  const int* __restrict__ diagonals,\n"
    "  double diagonal_value,\n"
    "  int num_rows,\n"
    "  const int* __restrict__ row_ptr,\n"
    "  const int* __restrict__ column_indices,\n"
    "  double* __restrict__ values)\n"
    "{\n"
    "  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "  for (size_t i = thread_idx;\n"
    "    i < num_diagonals;\n"
    "    i += blockDim.x * gridDim.x)\n"
    "  {\n"
    "    int row = diagonals[i];\n"
    "    int column = diagonals[i];\n"
    "    if (row >= num_rows) continue;\n"
    "    int r;\n"
    "    int err = binary_search(\n"
    "      row_ptr[row+1] - row_ptr[row],\n"
    "      &column_indices[row_ptr[row]],\n"
    "      column, &r);\n"
    "    if (err) continue;\n"
    "    r += row_ptr[row];\n"
    "    values[r] = diagonal_value;\n"
    "  }\n"
    "}\n";
}

/// CUDA C++ code for various useful device-side kernels
std::string cuda_kernel_assembly_utils(void)
{
  return
    cuda_kernel_zero_double_array() + "\n" +
    cuda_kernel_pack_coefficients() + "\n" +
    dolfinx::fem::cuda_kernel_binary_search() + "\n" +
    cuda_kernel_set_bc() + "\n" +
    cuda_kernel_set_diagonal();
}

void CUdeviceptr_free(CUdeviceptr* p)
{
  if (*p)
    cuMemFree(*p);
}

CUDA::Module compile_assembly_utils(
  const CUDA::Context& cuda_context,
  bool debug,
  const char* cudasrcdir,
  bool verbose)
{
  // Configure compiler options
  std::vector<std::string> compile_options = CUDA::nvrtc_compiler_options(cuda_context, debug);

  // Fetch the CUDA C++ code
  std::string assembly_utils_src =
    cuda_kernel_assembly_utils();

  // Compile CUDA C++ code to PTX assembly
  const char* program_name = "assembly_utils";
  int num_program_headers = 0;
  const char* program_headers[] = {};
  const char* program_include_names[] = {};
  std::string ptx = CUDA::compile_cuda_cpp_to_ptx(
    program_name, num_program_headers, program_headers,
    program_include_names, compile_options,
    assembly_utils_src.c_str(), cudasrcdir, verbose);

  // Load the PTX assembly as a module
  int num_module_load_options = 0;
  CUjit_option * module_load_options = NULL;
  void ** module_load_option_values = NULL;
  return CUDA::Module(
    cuda_context, ptx,
    num_module_load_options,
    module_load_options,
    module_load_option_values,
    verbose,
    debug);
}

} // namespace

//-----------------------------------------------------------------------------
CUDAAssembler::CUDAAssembler(
  const CUDA::Context& cuda_context,
  bool debug,
  const char* cudasrcdir,
  bool verbose)
  : _util_module(compile_assembly_utils(cuda_context, debug, cudasrcdir, verbose)),
    _verbose(verbose),
    _debug(debug)
{
}
//-----------------------------------------------------------------------------
void CUDAAssembler::zero_matrix_entries(
  const CUDA::Context& cuda_context,
  dolfinx::la::CUDAMatrix& A) const
{
  CUresult cuda_err;
  const char * cuda_err_description;

  // Fetch the device-side kernel
  CUfunction kernel = _util_module.get_device_function(
    "zero_double_array");

  int min_grid_size;
  int block_size;
  int shared_mem_size_per_thread_block = 0;
  cuda_err = cuOccupancyMaxPotentialBlockSize(
    &min_grid_size, &block_size, kernel, 0, 0, 0);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuOccupancyMaxPotentialBlockSize() failed with " +
      std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }

  unsigned int grid_dim_x = min_grid_size;
  unsigned int grid_dim_y = 1;
  unsigned int grid_dim_z = 1;
  unsigned int block_dim_x = block_size;
  unsigned int block_dim_y = 1;
  unsigned int block_dim_z = 1;
  unsigned int grid_size =
    grid_dim_x * grid_dim_y * grid_dim_z;
  unsigned int num_threads_per_block =
    block_dim_x * block_dim_y * block_dim_z;
  unsigned int num_threads =
    grid_size * num_threads_per_block;
  CUstream stream = NULL;

  // Launch device-side kernel
  {
    (void) cuda_context;
    dolfinx::la::CUDASeqMatrix * A_diag = A.diag();
    std::int32_t num_local_nonzeros = A_diag->num_local_nonzeros();
    CUdeviceptr dvalues = A_diag->values();
    void * kernel_parameters[] = {
      &num_local_nonzeros,
      &dvalues};
    cuda_err = cuLaunchKernel(
      kernel, grid_dim_x, grid_dim_y, grid_dim_z,
      block_dim_x, block_dim_y, block_dim_z,
      shared_mem_size_per_thread_block,
      stream, kernel_parameters, NULL);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuLaunchKernel() failed with " + std::string(cuda_err_description) +
        " at " + __FILE__ + ":" + std::to_string(__LINE__));
    }
  }

  {
    // Launch device-side kernel
    (void) cuda_context;
    dolfinx::la::CUDASeqMatrix * A_offdiag = A.offdiag();
    if (A_offdiag) {
      std::int32_t num_local_nonzeros = A_offdiag->num_local_nonzeros();
      CUdeviceptr dvalues = A_offdiag->values();
      void * kernel_parameters[] = {
        &num_local_nonzeros,
        &dvalues};
      cuda_err = cuLaunchKernel(
        kernel, grid_dim_x, grid_dim_y, grid_dim_z,
        block_dim_x, block_dim_y, block_dim_z,
        shared_mem_size_per_thread_block,
        stream, kernel_parameters, NULL);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuLaunchKernel() failed with " + std::string(cuda_err_description) +
          " at " + __FILE__ + ":" + std::to_string(__LINE__));
      }
    }
  }

  // Wait for the kernel to finish.
  cuda_err = cuCtxSynchronize();
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuCtxSynchronize() failed with " + std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }
}
//-----------------------------------------------------------------------------
void CUDAAssembler::zero_vector_entries(
  const CUDA::Context& cuda_context,
  dolfinx::la::CUDAVector& x) const
{
  CUresult cuda_err;
  const char * cuda_err_description;

  // Fetch the device-side kernel
  CUfunction kernel = _util_module.get_device_function(
    "zero_double_array");

  int min_grid_size;
  int block_size;
  int shared_mem_size_per_thread_block = 0;
  cuda_err = cuOccupancyMaxPotentialBlockSize(
    &min_grid_size, &block_size, kernel, 0, 0, 0);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuOccupancyMaxPotentialBlockSize() failed with " +
      std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }

  unsigned int grid_dim_x = min_grid_size;
  unsigned int grid_dim_y = 1;
  unsigned int grid_dim_z = 1;
  unsigned int block_dim_x = block_size;
  unsigned int block_dim_y = 1;
  unsigned int block_dim_z = 1;
  unsigned int grid_size =
    grid_dim_x * grid_dim_y * grid_dim_z;
  unsigned int num_threads_per_block =
    block_dim_x * block_dim_y * block_dim_z;
  unsigned int num_threads =
    grid_size * num_threads_per_block;
  CUstream stream = NULL;

  // Launch device-side kernel
  (void) cuda_context;
  std::int32_t num_local_nonzeros =
      x.ghosted() ? x.num_local_ghosted_values() : x.num_local_values();
  CUdeviceptr dvalues = x.values_write();
  void * kernel_parameters[] = {
    &num_local_nonzeros,
    &dvalues};
  cuda_err = cuLaunchKernel(
    kernel, grid_dim_x, grid_dim_y, grid_dim_z,
    block_dim_x, block_dim_y, block_dim_z,
    shared_mem_size_per_thread_block,
    stream, kernel_parameters, NULL);
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuLaunchKernel() failed with " + std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }

  // Wait for the kernel to finish.
  cuda_err = cuCtxSynchronize();
  if (cuda_err != CUDA_SUCCESS) {
    cuGetErrorString(cuda_err, &cuda_err_description);
    throw std::runtime_error(
      "cuCtxSynchronize() failed with " + std::string(cuda_err_description) +
      " at " + __FILE__ + ":" + std::to_string(__LINE__));
  }

  x.restore_values_write();
}
