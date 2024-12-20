// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of cuDOLFINX
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cuda.h>
#include <cudolfinx/common/CUDA.h>
#include <cudolfinx/mesh/CUDAMesh.h>
#include <cudolfinx/fem/CUDADofMap.h>
#include <cudolfinx/fem/CUDAFormConstants.h>
#include <cudolfinx/fem/CUDAFormCoefficients.h>
#include <cudolfinx/fem/CUDAFormIntegral.h>
#include <cudolfinx/fem/CUDADirichletBC.h>
#include <cudolfinx/fem/CUDADofMap.h>
#include <cudolfinx/la/CUDAMatrix.h>
#include <cudolfinx/la/CUDASeqMatrix.h>
#include <cudolfinx/la/CUDAVector.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>



#include <map>
#include <memory>
#include <vector>
#include <utility>

struct ufc_integral;

namespace dolfinx
{

namespace CUDA
{
class Context;
class Module;
}

namespace function
{
class FunctionSpace;
}


namespace la
{
class CUDAMatrix;
class CUDAVector;
};

namespace fem
{

/// Interface for GPU-accelerated assembly of variational forms.
class CUDAAssembler
{
public:
  /// Create CUDA-based assembler
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] debug Whether or not to compile CUDA C++ code with
  ///                  debug information
  /// @param[in] cudasrcdir Path for outputting CUDA C++ code
  CUDAAssembler(
    const CUDA::Context& cuda_context,
    CUjit_target target,
    bool debug,
    const char* cudasrcdir,
    bool verbose);

  /// Destructor
  ~CUDAAssembler() = default;

  /// Copy constructor
  /// @param[in] assembler The object to be copied
  CUDAAssembler(const CUDAAssembler& assembler) = delete;

  /// Move constructor
  /// @param[in] assembler The object to be moved
  CUDAAssembler(CUDAAssembler&& assembler) = delete;

  /// Assignment operator
  /// @param[in] assembler Another CUDAAssembler object
  CUDAAssembler& operator=(const CUDAAssembler& assembler) = delete;

  /// Move assignment operator
  /// @param[in] assembler Another CUDAAssembler object
  CUDAAssembler& operator=(CUDAAssembler&& assembler) = delete;

  /// Set the entries of a device-side CSR matrix to zero
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] A The device-side CSR matrix
  void zero_matrix_entries(
    const CUDA::Context& cuda_context,
    dolfinx::la::CUDAMatrix& A) const;

  /// Set the entries of a device-side vector to zero
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] x The device-side vector
  void zero_vector_entries(
    const CUDA::Context& cuda_context,
    dolfinx::la::CUDAVector& x) const;


  //-----------------------------------------------------------------------------
  /// Pack coefficient values for a form.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] coefficients Device-side data for form coefficients
  template <dolfinx::scalar T,
           std::floating_point U = dolfinx::scalar_value_type_t<T>>
  void pack_coefficients(
    const CUDA::Context& cuda_context,
    dolfinx::fem::CUDAFormCoefficients<T,U>& coefficients
   ) const
  {
    std::vector<int> indices;
    for (int i = 0; i < coefficients.num_coefficients(); i++)
      indices.push_back(i);

    repack_coefficients(cuda_context, coefficients, indices);
  }

  //-----------------------------------------------------------------------------
  /// Pack a subset of coefficient values for a form.
  /// This function is used as an optimization for cases when only a subset of coefficients
  /// have changed.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] coefficients Device-side data for form coefficients
  /// @param[in] coefficients_to_pack 
  template <dolfinx::scalar T,
           std::floating_point U = dolfinx::scalar_value_type_t<T>>
  void pack_coefficients(
    const CUDA::Context& cuda_context,
    dolfinx::fem::CUDAFormCoefficients<T,U>& coefficients,
    std::vector<std::shared_ptr<dolfinx::fem::Function<T,U>>> coefficients_to_pack
   ) const
  {
    if (!coefficients_to_pack.size()) return;

    auto form_coeffs = coefficients.coefficients();

    if (coefficients_to_pack.size() > form_coeffs.size()) {
      throw std::runtime_error("Too many coefficients to pack!");
    }

    std::vector<int> indices;

    for (int i = 0; i < coefficients_to_pack.size(); i++) {
      for (int j = 0; j < form_coeffs.size(); j++) {
        if (form_coeffs[j] == coefficients_to_pack[i]) {
          indices.push_back(j);
          break;
        }
      }
    }

    if (indices.size() != coefficients_to_pack.size()) {
      throw std::runtime_error("Unable to match all coefficients to existing coefficients in the Form!");
    }

    repack_coefficients(cuda_context, coefficients, indices);
  }

  //-----------------------------------------------------------------------------
  /// (Re-)pack coefficient values for a form.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] coefficients Device-side data for form coefficients
  /// @param[in] indices Indices of coefficients to repack
  template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
  void repack_coefficients(
    const CUDA::Context& cuda_context,
    dolfinx::fem::CUDAFormCoefficients<T,U>& coefficients,
    std::vector<int>& indices) const
  {
    CUresult cuda_err;
    const char * cuda_err_description;

    {
        std::vector<CUdeviceptr> coefficient_values = coefficients.coefficient_device_ptrs();
        std::int32_t num_coefficients = coefficient_values.size();
        CUdeviceptr dcoefficient_values = coefficients.coefficient_values();
        size_t coefficient_values_size = num_coefficients * sizeof(CUdeviceptr);
        cuda_err = cuMemcpyHtoD(
            dcoefficient_values, coefficient_values.data(), coefficient_values_size);
        if (cuda_err != CUDA_SUCCESS) {
            cuGetErrorString(cuda_err, &cuda_err_description);
            throw std::runtime_error(
                "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
                " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
        }

        CUdeviceptr dcoefficient_indices = coefficients.coefficient_indices();
        CUDA::safeMemcpyHtoD(dcoefficient_indices, indices.data(), indices.size() * sizeof(int));
    }

    // Fetch the device-side kernel
    CUfunction kernel = _util_module.get_device_function(
      "pack_coefficients");

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
    CUdeviceptr dofmaps_num_dofs_per_cell = coefficients.dofmaps_num_dofs_per_cell();
    CUdeviceptr dofmaps_dofs_per_cell = coefficients.dofmaps_dofs_per_cell();
    CUdeviceptr coefficient_values_offsets = coefficients.coefficient_values_offsets();
    std::int32_t num_coefficients = coefficients.num_coefficients();
    CUdeviceptr coefficient_values = coefficients.coefficient_values();
    CUdeviceptr coefficient_indices = coefficients.coefficient_indices();
    std::int32_t num_indices = indices.size();
    std::int32_t num_cells = coefficients.num_cells();
    std::int32_t num_packed_coefficient_values_per_cell =
      coefficients.num_packed_coefficient_values_per_cell();
    CUdeviceptr packed_coefficient_values = coefficients.packed_coefficient_values();
    void * kernel_parameters[] = {
        &num_coefficients,
        &dofmaps_num_dofs_per_cell,
        &dofmaps_dofs_per_cell,
        &coefficient_values_offsets,
        &coefficient_values,
        &coefficient_indices,
        &num_indices,
        &num_cells,
        &num_packed_coefficient_values_per_cell,
        &packed_coefficient_values};

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

  }
  //-----------------------------------------------------------------------------
  /// Assemble linear form into a vector. The vector must already be
  /// initialised. Does not zero vector and ghost contributions are
  /// not accumulated (sent to owner). Caller is responsible for
  /// calling VecGhostUpdateBegin/End.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] mesh Device-side mesh data
  /// @param[in] dofmap Device-side data for degrees of freedom
  /// @param[in] bc Device-side data for Dirichlet boundary conditions
  /// @param[in] form_integrals Device-side kernels and data for each
  ///                           integral of the variational form
  /// @param[in] constants Device-side data for form constants
  /// @param[in] coefficients Device-side data for form coefficients
  /// @param[in,out] b The device-side vector to assemble the form into
  template <dolfinx::scalar T,
        std::floating_point U = dolfinx::scalar_value_type_t<T>>
  void assemble_vector(
  const CUDA::Context& cuda_context,
  const dolfinx::mesh::CUDAMesh<U>& mesh,
  const dolfinx::fem::CUDADofMap& dofmap,
//  const dolfinx::fem::CUDADirichletBC<T,U>& bc,
  const std::map<IntegralType, std::vector<CUDAFormIntegral<T,U>>>& form_integrals,
  const dolfinx::fem::CUDAFormConstants<T>& constants,
  const dolfinx::fem::CUDAFormCoefficients<T,U>& coefficients,
  dolfinx::la::CUDAVector& b,
  bool verbose) const
  {
    {
      // Perform assembly for cell integrals
      auto it = form_integrals.find(IntegralType::cell);
      if (it != form_integrals.end()) {
        const std::vector<CUDAFormIntegral<T,U>>& cuda_cell_integrals = it->second;
        for (auto const& cuda_cell_integral : cuda_cell_integrals) {
          cuda_cell_integral.assemble_vector(
            cuda_context, mesh, dofmap,
            constants, coefficients, b, verbose);
        }
      }
    }

    {
      // Perform assembly for exterior facet integrals
      auto it = form_integrals.find(IntegralType::exterior_facet);
      if (it != form_integrals.end()) {
        const std::vector<CUDAFormIntegral<T,U>>&
          cuda_exterior_facet_integrals = it->second;
        for (auto const& cuda_exterior_facet_integral :
               cuda_exterior_facet_integrals)
        {
          cuda_exterior_facet_integral.assemble_vector(
            cuda_context, mesh, dofmap,
            constants, coefficients, b, verbose);
        }
      }
    }

    {
      // Perform assembly for interior facet integrals
      auto it = form_integrals.find(IntegralType::interior_facet);
      if (it != form_integrals.end()) {
        const std::vector<CUDAFormIntegral<T,U>>&
          cuda_interior_facet_integrals = it->second;
        for (auto const& cuda_interior_facet_integral :
               cuda_interior_facet_integrals)
        {
          cuda_interior_facet_integral.assemble_vector(
            cuda_context, mesh, dofmap,
            constants, coefficients, b, verbose);
        }
      }
    }
  }
  //-----------------------------------------------------------------------------
  /// Apply Dirichlet boundary conditions to a vector.
  ///
  /// For points where the given boundary conditions apply, the value
  /// is set to
  ///
  ///   b <- b - scale * (g - x0),
  ///
  /// where g denotes the boundary condition values. The boundary
  /// conditions bcs are defined on the test space or a subspace of
  /// the test space of the given linear form that was used to
  /// assemble the vector.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] bcs Device-side data for Dirichlet boundary conditions
  /// @param[in] x0 A device-side vector
  /// @param[in] scale Scaling factor
  /// @param[in,out] b The device-side vector to modify
  template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
  void set_bc(
    const CUDA::Context& cuda_context,
    const dolfinx::fem::CUDADirichletBC<T,U>& bc,
    std::shared_ptr<dolfinx::la::CUDAVector> x0,
    double scale,
    dolfinx::la::CUDAVector& b) const
  {
    CUresult cuda_err;
    const char * cuda_err_description;

    // Fetch the device-side kernel
    CUfunction kernel = _util_module.get_device_function(
      "set_bc");

    // Compute a block size with high occupancy
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
    std::int32_t num_boundary_dofs = bc.num_boundary_dofs();
    CUdeviceptr dboundary_dofs = bc.dof_indices();
    CUdeviceptr dboundary_value_dofs = bc.dof_value_indices();
    CUdeviceptr dboundary_values = bc.dof_values();
    CUdeviceptr dx0 = (x0) ? x0->values() : NULL;
    std::int32_t num_values =
        b.ghosted() ? b.num_local_ghosted_values() : b.num_local_values();
    CUdeviceptr dvalues = b.values_write();
    void * kernel_parameters[] = {
      &num_boundary_dofs,
      &dboundary_dofs,
      &dboundary_value_dofs,
      &dboundary_values,
      &dx0,
      &scale,
      &num_values,
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

    b.restore_values_write();
    if (x0) x0->restore_values();
  }
  //-----------------------------------------------------------------------------
  /// Modify a right-hand side vector `b` to account for essential
  /// boundary conditions,
  ///
  ///   b <- b - scale * A (g - x0),
  ///
  /// where `g` denotes the boundary condition values. The boundary
  /// conditions `bcs1` are defined on the trial space of the given
  /// bilinear form. The test space of the bilinear form must be the
  /// same as the test space of the linear form from which `b` is
  /// assembled, but the trial spaces may differ.
  ///
  /// Ghost contributions are not accumulated (not sent to owner). The
  /// caller is responsible for calling `VecGhostUpdateBegin/End()`.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] mesh Device-side mesh data
  /// @param[in] dofmap0 Device-side data for degrees of freedom of
  ///                    the test space
  /// @param[in] dofmap1 Device-side data for degrees of freedom of
  ///                    the trial space
  /// @param[in] form_integrals Device-side kernels and data for each
  ///                           integral of the variational form. Note
  ///                           that this refers to the form that was
  ///                           used to assemble the coefficient matrix.
  /// @param[in] constants Device-side data for form constants
  /// @param[in] coefficients Device-side data for form coefficients
  /// @param[in] bcs1 Device-side data for Dirichlet boundary
  ///                 conditions on the trial space
  /// @param[in] x0 A device-side vector
  /// @param[in] scale Scaling factor
  /// @param[in,out] b The device-side vector to modify
  template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
  void lift_bc(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::CUDAMesh<U>& mesh,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const dolfinx::fem::CUDADofMap& dofmap1,
    const std::map<IntegralType, std::vector<CUDAFormIntegral<T,U>>>& form_integrals,
    const dolfinx::fem::CUDAFormConstants<T>& constants,
    const dolfinx::fem::CUDAFormCoefficients<T,U>& coefficients,
    const dolfinx::fem::CUDADirichletBC<T,U>& bc1,
    std::shared_ptr<dolfinx::la::CUDAVector> x0,
    double scale,
    dolfinx::la::CUDAVector& b,
    bool verbose) const
  {
    {
      // Apply boundary conditions for cell integrals
      auto it = form_integrals.find(IntegralType::cell);
      if (it != form_integrals.end()) {
        const std::vector<CUDAFormIntegral<T,U>>& cuda_cell_integrals = it->second;
        auto const& cuda_cell_integral = cuda_cell_integrals.at(0);
        cuda_cell_integral.lift_bc(
          cuda_context, mesh, dofmap0, dofmap1, bc1,
          constants, coefficients, scale, x0, b, verbose);
      }
    }

    {
      // Apply boundary conditions for exterior facet integrals
      auto it = form_integrals.find(IntegralType::exterior_facet);
      if (it != form_integrals.end()) {
        const std::vector<CUDAFormIntegral<T,U>>& cuda_exterior_facet_integrals =
          it->second;
        auto const& cuda_exterior_facet_integral =
          cuda_exterior_facet_integrals.at(0);
        cuda_exterior_facet_integral.lift_bc(
          cuda_context, mesh, dofmap0, dofmap1, bc1,
          constants, coefficients, scale, x0, b, verbose);
      }
    }
  }
  //-----------------------------------------------------------------------------
  /// Modify a right-hand side vector `b` to account for essential
  /// boundary conditions,
  ///
  ///   b <- b - scale * A_j (g_j - x0_j),
  ///
  /// where `g_j` denotes the boundary condition values and `j` is
  /// block (nest) index, or `j=0` if the problem is not blocked. The
  /// boundary conditions `bcs1` are defined on the trial spaces `V_j`
  /// of the `j`-th bilinear form. The test spaces of the bilinear
  /// forms must be the same as the test space of the linear form from
  /// which `b` is assembled, but the trial spaces may differ.
  ///
  /// Ghost contributions are not accumulated (not sent to owner). The
  /// caller is responsible for calling `VecGhostUpdateBegin/End()`.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] mesh Device-side mesh data
  /// @param[in] dofmap0 Device-side data for degrees of freedom of
  ///                    the test space
  /// @param[in] dofmap1 Device-side data for degrees of freedom of
  ///                    the trial space
  /// @param[in] form_integrals Device-side kernels and data for each
  ///                           integral of the variational form. Note
  ///                           that this refers to the form that was
  ///                           used to assemble the coefficient matrix.
  /// @param[in] constants Device-side data for form constants
  /// @param[in] coefficients Device-side data for form coefficients
  /// @param[in] bcs1 Device-side data for Dirichlet boundary
  ///                 conditions on the trial spaces
  /// @param[in] x0 A device-side vector
  /// @param[in] scale Scaling factor
  /// @param[in,out] b The device-side vector to modify
/*  template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
  void apply_lifting(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::CUDAMesh<U>& mesh,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const std::vector<const dolfinx::fem::CUDADofMap*>& dofmap1,
    const std::vector<const std::map<IntegralType, std::vector<CUDAFormIntegral<T,U>>>*>& form_integrals,
    const std::vector<const dolfinx::fem::CUDAFormConstants<T>*>& constants,
    const std::vector<const dolfinx::fem::CUDAFormCoefficients<T,U>*>& coefficients,
    const std::vector<const dolfinx::fem::CUDADirichletBC<T,U>*>& bcs1,
    const std::vector<const dolfinx::la::CUDAVector*>& x0,
    double scale,
    dolfinx::la::CUDAVector& b,
    bool verbose) const
  {
    for (int i = 0; i < form_integrals.size(); i++) {
      lift_bc(cuda_context, mesh, dofmap0, *dofmap1[i],
              *form_integrals[i], *constants[i], *coefficients[i],
              *bcs1[i], *x0[i], scale, b, verbose);
    }
  }*/
  //-----------------------------------------------------------------------------
  /// Assemble bilinear form into a matrix. Matrix must already be
  /// initialised. Does not zero or finalise the matrix.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] mesh Device-side mesh data
  /// @param[in] dofmap0 Device-side data for degrees of freedom of
  ///                    the test space
  /// @param[in] dofmap1 Device-side data for degrees of freedom of
  ///                    the trial space
  /// @param[in] bc0 Device-side data for Dirichlet boundary
  ///                conditions on the test space
  /// @param[in] bc1 Device-side data for Dirichlet boundary
  ///                conditions on the trial space
  /// @param[in] form_integrals Device-side kernels and data for each
  ///                           integral of the variational form
  /// @param[in] constants Device-side data for form constants
  /// @param[in] coefficients Device-side data for form coefficients
  /// @param[in,out] A The device-side CSR matrix that is used to
  ///                  store the assembled form. The matrix must be
  ///                  initialised before calling this function. The
  ///                  matrix is not zeroed.
  template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
  void assemble_matrix(
    const CUDA::Context& cuda_context,
    const dolfinx::mesh::CUDAMesh<U>& mesh,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const dolfinx::fem::CUDADofMap& dofmap1,
    const dolfinx::fem::CUDADirichletBC<T,U>& bc0,
    const dolfinx::fem::CUDADirichletBC<T,U>& bc1,
    std::map<IntegralType, std::vector<CUDAFormIntegral<T,U>>>& form_integrals,
    const dolfinx::fem::CUDAFormConstants<T>& constants,
    const dolfinx::fem::CUDAFormCoefficients<T,U>& coefficients,
    dolfinx::la::CUDAMatrix& A,
    bool verbose) const
  {
    {
      // Perform assembly for cell integrals
      auto it = form_integrals.find(IntegralType::cell);
      if (it != form_integrals.end()) {
        std::vector<CUDAFormIntegral<T,U>>& cuda_cell_integrals = it->second;
        for (auto & cuda_cell_integral : cuda_cell_integrals) {
          cuda_cell_integral.assemble_matrix(
            cuda_context, mesh, dofmap0, dofmap1, bc0, bc1,
            constants, coefficients, A, verbose);
        }
      }
    }

    {
      // Perform assembly for exterior facet integrals
      auto it = form_integrals.find(IntegralType::exterior_facet);
      if (it != form_integrals.end()) {
        std::vector<CUDAFormIntegral<T,U>>&
          cuda_exterior_facet_integrals = it->second;
        for (auto & cuda_exterior_facet_integral :
               cuda_exterior_facet_integrals)
        {
          cuda_exterior_facet_integral.assemble_matrix(
            cuda_context, mesh, dofmap0, dofmap1, bc0, bc1,
            constants, coefficients, A, verbose);
        }
      }
    }

    {
      // Perform assembly for interior facet integrals
      auto it = form_integrals.find(IntegralType::interior_facet);
      if (it != form_integrals.end()) {
        std::vector<CUDAFormIntegral<T,U>>&
          cuda_interior_facet_integrals = it->second;
        for (auto & cuda_interior_facet_integral :
               cuda_interior_facet_integrals)
        {
          cuda_interior_facet_integral.assemble_matrix(
            cuda_context, mesh, dofmap0, dofmap1, bc0, bc1,
            constants, coefficients, A, verbose);
        }
      }
    }
  }
  //-----------------------------------------------------------------------------
  /// Copy element matrices from CUDA device memory to host memory.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] form_integrals Device-side kernels and data for each
  ///                           integral of the variational form
  template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
  void assemble_matrix_local_copy_to_host(
    const CUDA::Context& cuda_context,
    std::map<IntegralType, std::vector<CUDAFormIntegral<T,U>>>& form_integrals) const
  {
    {
      // Perform assembly for cell integrals
      auto it = form_integrals.find(IntegralType::cell);
      if (it != form_integrals.end()) {
        std::vector<CUDAFormIntegral<T,U>>& cuda_cell_integrals = it->second;
        for (auto & cuda_cell_integral : cuda_cell_integrals) {
          cuda_cell_integral.assemble_matrix_local_copy_to_host(
            cuda_context);
        }
      }
    }

    {
      // Perform assembly for exterior facet integrals
      auto it = form_integrals.find(IntegralType::exterior_facet);
      if (it != form_integrals.end()) {
        std::vector<CUDAFormIntegral<T,U>>&
          cuda_exterior_facet_integrals = it->second;
        for (auto & cuda_exterior_facet_integral :
               cuda_exterior_facet_integrals)
        {
          cuda_exterior_facet_integral.assemble_matrix_local_copy_to_host(
            cuda_context);
        }
      }
    }

    {
      // Perform assembly for interior facet integrals
      auto it = form_integrals.find(IntegralType::interior_facet);
      if (it != form_integrals.end()) {
        std::vector<CUDAFormIntegral<T,U>>&
          cuda_interior_facet_integrals = it->second;
        for (auto & cuda_interior_facet_integral :
               cuda_interior_facet_integrals)
        {
          cuda_interior_facet_integral.assemble_matrix_local_copy_to_host(
            cuda_context);
        }
      }
    }
  }
  //-----------------------------------------------------------------------------
  /// Perform global assembly on the host.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] dofmap0 Device-side data for degrees of freedom of
  ///                    the test space
  /// @param[in] dofmap1 Device-side data for degrees of freedom of
  ///                    the trial space
  /// @param[in] form_integrals Device-side kernels and data for each
  ///                           integral of the variational form
  /// @param[in,out] A The device-side CSR matrix that is used to
  ///                  store the assembled form. The matrix must be
  ///                  initialised before calling this function. The
  ///                  matrix is not zeroed.
  template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
  void assemble_matrix_local_host_global_assembly(
    const CUDA::Context& cuda_context,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const dolfinx::fem::CUDADofMap& dofmap1,
    std::map<IntegralType, std::vector<CUDAFormIntegral<T,U>>>& form_integrals,
    dolfinx::la::CUDAMatrix& A) const
  {
    {
      // Perform assembly for cell integrals
      auto it = form_integrals.find(IntegralType::cell);
      if (it != form_integrals.end()) {
        std::vector<CUDAFormIntegral<T,U>>& cuda_cell_integrals = it->second;
        for (auto & cuda_cell_integral : cuda_cell_integrals) {
          cuda_cell_integral.assemble_matrix_local_host_global_assembly(
            cuda_context, dofmap0, dofmap1, A);
        }
      }
    }

    {
      // Perform assembly for exterior facet integrals
      auto it = form_integrals.find(IntegralType::exterior_facet);
      if (it != form_integrals.end()) {
        std::vector<CUDAFormIntegral<T,U>>&
          cuda_exterior_facet_integrals = it->second;
        for (auto & cuda_exterior_facet_integral :
               cuda_exterior_facet_integrals)
        {
          cuda_exterior_facet_integral.assemble_matrix_local_host_global_assembly(
            cuda_context, dofmap0, dofmap1, A);
        }
      }
    }

    {
      // Perform assembly for interior facet integrals
      auto it = form_integrals.find(IntegralType::interior_facet);
      if (it != form_integrals.end()) {
        std::vector<CUDAFormIntegral<T,U>>&
          cuda_interior_facet_integrals = it->second;
        for (auto & cuda_interior_facet_integral :
               cuda_interior_facet_integrals)
        {
          cuda_interior_facet_integral.assemble_matrix_local_host_global_assembly(
            cuda_context, dofmap0, dofmap1, A);
        }
      }
    }
  }

  //-----------------------------------------------------------------------------
  /// Set a value on the diagonal entries of a matrix that belong to
  /// rows where a Dirichlet boundary condition applies. This function
  /// is typically called after assembly. The assembly function does
  /// not add any contributions to rows and columns that are affected
  /// by Dirichlet boundary conditions. This function sets a value
  /// only for rows that are locally owned, and therefore does not
  /// create a need for parallel communication. For block matrices,
  /// this function should normally be called only on the diagonal
  /// blocks, that is, blocks for which the test and trial spaces are
  /// the same.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in,out] A The matrix to set diagonal values for
  /// @param[in] bc The Dirichlet boundary condtions
  /// @param[in] diagonal The value to set on the diagonal for rows with a
  ///                     boundary condition applied
  template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
  void set_diagonal(
    const CUDA::Context& cuda_context,
    dolfinx::la::CUDAMatrix& A,
    const dolfinx::fem::CUDADirichletBC<T,U>& bc,
    double diagonal = 1.0) const
  {
    CUresult cuda_err;
    const char * cuda_err_description;

    // Fetch the device-side kernel
    CUfunction kernel = _util_module.get_device_function(
      "set_diagonal");

    // Compute a block size with high occupancy
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
    std::int32_t num_diagonals = bc.num_owned_boundary_dofs();
    CUdeviceptr ddiagonals = bc.dof_indices();
    std::int32_t num_rows = A.diag()->num_rows();
    CUdeviceptr drow_ptr = A.diag()->row_ptr();
    CUdeviceptr dcolumn_indices = A.diag()->column_indices();
    CUdeviceptr dvalues = A.diag()->values();
    void * kernel_parameters[] = {
      &num_diagonals,
      &ddiagonals,
      &diagonal,
      &num_rows,
      &drow_ptr,
      &dcolumn_indices,
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
  }
  //-----------------------------------------------------------------------------
  /// Compute lookup tables that are used during matrix assembly for
  /// kernels that need it
  template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
  void compute_lookup_tables(
    const CUDA::Context& cuda_context,
    const dolfinx::fem::CUDADofMap& dofmap0,
    const dolfinx::fem::CUDADofMap& dofmap1,
    const dolfinx::fem::CUDADirichletBC<T,U>& bc0,
    const dolfinx::fem::CUDADirichletBC<T,U>& bc1,
    std::map<IntegralType, std::vector<CUDAFormIntegral<T,U>>>& form_integrals,
    dolfinx::la::CUDAMatrix& A,
    bool verbose) const
  {
    {
      auto it = form_integrals.find(IntegralType::cell);
      if (it != form_integrals.end()) {
        std::vector<CUDAFormIntegral<T,U>>& cuda_cell_integrals = it->second;
        auto & cuda_cell_integral = cuda_cell_integrals.at(0);
        cuda_cell_integral.compute_lookup_table(
          cuda_context, dofmap0, dofmap1, bc0, bc1, A, verbose);
      }
    }

    {
      auto it = form_integrals.find(IntegralType::exterior_facet);
      if (it != form_integrals.end()) {
        std::vector<CUDAFormIntegral<T,U>>& cuda_exterior_facet_integrals =
          it->second;
        auto & cuda_exterior_facet_integral =
          cuda_exterior_facet_integrals.at(0);
        cuda_exterior_facet_integral.compute_lookup_table(
          cuda_context, dofmap0, dofmap1, bc0, bc1, A, verbose);
      }
    }

    {
      auto it = form_integrals.find(IntegralType::interior_facet);
      if (it != form_integrals.end()) {
        std::vector<CUDAFormIntegral<T,U>>& cuda_interior_facet_integrals =
          it->second;
        auto & cuda_interior_facet_integral =
          cuda_interior_facet_integrals.at(0);
        cuda_interior_facet_integral.compute_lookup_table(
          cuda_context, dofmap0, dofmap1, bc0, bc1, A, verbose);
      }
    }
  }
  //-----------------------------------------------------------------------------



private:
  /// Module for various useful device-side functions
  CUDA::Module _util_module;
};

} // namespace fem
} // namespace dolfinx
