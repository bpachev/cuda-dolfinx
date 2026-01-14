// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of cuDOLFINX
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cudolfinx/common/CUDA.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/common/IndexMap.h>
#include <cuda.h>

#include <memory>
#include <vector>

namespace dolfinx {

//namespace function {
//class FunctionSpace;
//}

namespace fem {
//class DirichletBC;

/// A wrapper for data marking which degrees of freedom that are
/// affected by Dirichlet boundary conditions, with data being stored
/// in the device memory of a CUDA device.
template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_t<T>>
class CUDADirichletBC
{
public:

//-----------------------------------------------------------------------------
  /// Create empty Dirichlet boundary conditions
  CUDADirichletBC()
    : _num_dofs()
    , _num_owned_boundary_dofs()
    , _num_boundary_dofs()
    , _ddof_markers(0)
    , _ddof_indices(0)
    , _ddof_values(0)
  {
  }
  //-----------------------------------------------------------------------------
  /// Create Dirichlet boundary conditions
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] V The function space to build dof markers for.
  ///              Boundary conditions are only applied for degrees of
  ///              freedom that belong to the given function space.
  /// @param[in] bcs The boundary conditions to copy to device memory
  CUDADirichletBC(
    const CUDA::Context& cuda_context,
    const dolfinx::fem::FunctionSpace<T>& V,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T,U>>>& bcs)
    : _num_dofs()
    , _num_owned_boundary_dofs()
    , _num_boundary_dofs()
    , _ddof_markers(0)
    , _ddof_indices(0)
    , _ddof_values(0)
  {
    CUresult cuda_err;
    const char * cuda_err_description;

    // Count the number of degrees of freedom
    const dolfinx::fem::DofMap& dofmap = *(V.dofmap());
    const common::IndexMap& index_map = *dofmap.index_map;
    // Looks like index_map no longer has block_size
    const int block_size = dofmap.index_map_bs();
    _num_dofs = block_size * (
		    index_map.size_local() + index_map.num_ghosts());

    // Count the number of degrees of freedom affected by boundary
    // conditions
    _num_owned_boundary_dofs = 0;
    _num_boundary_dofs = 0;

    // Build dof markers, indices and values
    signed char* dof_markers = nullptr;
    std::vector<std::int32_t> dof_indices;
    std::vector<std::int32_t> ghost_dof_indices;
    for (auto const& bc : bcs) {
      if (V.contains(*bc->function_space())) {
        if (!dof_markers) {
          dof_markers = new signed char[_num_dofs];
          for (int i = 0; i < _num_dofs; i++) {
            dof_markers[i] = 0;
          }
          _dof_values.assign(_num_dofs, 0.0);
        }
        
        bc->mark_dofs(std::span(dof_markers, _num_dofs));
        auto const [dofs, range] = bc->dof_indices();
        for (std::int32_t i = 0; i < dofs.size(); i++) {
	  if (i < range) dof_indices.push_back(dofs[i]);
	  else ghost_dof_indices.push_back(dofs[i]);
        }
        bc->set(std::span<T>(_dof_values), {}, 1);
      }
    }
    _num_owned_boundary_dofs = dof_indices.size();
    _num_boundary_dofs = _num_owned_boundary_dofs + ghost_dof_indices.size();
    dof_indices.insert(dof_indices.end(), ghost_dof_indices.begin(), ghost_dof_indices.end());
    // Allocate device-side storage for dof markers
    if (dof_markers && _num_dofs > 0) {
      size_t ddof_markers_size = _num_dofs * sizeof(char);
      cuda_err = cuMemAlloc(&_ddof_markers, ddof_markers_size);
      if (cuda_err != CUDA_SUCCESS) {
        delete[] dof_markers;
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemAlloc() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }

      // Copy dof markers to device
      cuda_err = cuMemcpyHtoD(
        _ddof_markers, dof_markers, ddof_markers_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuMemFree(_ddof_markers);
        delete[] dof_markers;
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
    }
    if (dof_markers)
      delete[] dof_markers;

    // Allocate device-side storage for dof indices
    if (_num_boundary_dofs > 0) {
      size_t ddof_indices_size = dof_indices.size() * sizeof(std::int32_t);
      cuda_err = cuMemAlloc(&_ddof_indices, ddof_indices_size);
      if (cuda_err != CUDA_SUCCESS) {
        if (_ddof_markers)
          cuMemFree(_ddof_markers);
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemAlloc() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }

      // Copy dof indices to device
      cuda_err = cuMemcpyHtoD(
        _ddof_indices, dof_indices.data(), ddof_indices_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuMemFree(_ddof_indices);
        if (_ddof_markers)
          cuMemFree(_ddof_markers);
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
    }


    // Allocate device-side storage for dof values
    if (dof_markers && _num_dofs > 0) {
      size_t ddof_values_size = _num_dofs * sizeof(T);
      cuda_err = cuMemAlloc(&_ddof_values, ddof_values_size);
      if (cuda_err != CUDA_SUCCESS) {
        if (_ddof_indices)
          cuMemFree(_ddof_indices);
        if (_ddof_markers)
          cuMemFree(_ddof_markers);
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemAlloc() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }

      // Copy dof values to device
      cuda_err = cuMemcpyHtoD(
        _ddof_values, _dof_values.data(), ddof_values_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuMemFree(_ddof_values);
        if (_ddof_indices)
          cuMemFree(_ddof_indices);
        if (_ddof_markers)
          cuMemFree(_ddof_markers);
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
    }
  }
  //-----------------------------------------------------------------------------
  /// Destructor
  ~CUDADirichletBC()
  {
    if (_ddof_values)
      cuMemFree(_ddof_values);
    if (_ddof_indices)
      cuMemFree(_ddof_indices);
    if (_ddof_markers)
      cuMemFree(_ddof_markers);
  }
  //-----------------------------------------------------------------------------
  /// Copy constructor
  /// @param[in] bc The object to be copied
  CUDADirichletBC(const CUDADirichletBC& bc) = delete;

  /// Move constructor
  /// @param[in] bc The object to be moved
  CUDADirichletBC(CUDADirichletBC&& bc)
    : _num_dofs(bc._num_dofs)
    , _num_owned_boundary_dofs(bc._num_owned_boundary_dofs)
    , _num_boundary_dofs(bc._num_boundary_dofs)
    , _ddof_markers(bc._ddof_markers)
    , _ddof_indices(bc._ddof_indices)
    , _ddof_values(bc._ddof_values)
  {
    bc._num_dofs = 0;
    bc._num_owned_boundary_dofs = 0;
    bc._num_boundary_dofs = 0;
    bc._ddof_markers = 0;
    bc._ddof_indices = 0;
    bc._ddof_values = 0;
  }
  //-----------------------------------------------------------------------------
  /// Assignment operator
  /// @param[in] bc Another CUDADirichletBC object
  CUDADirichletBC& operator=(const CUDADirichletBC& bc) = delete;

  /// Move assignment operator
  /// @param[in] bc Another CUDADirichletBC object
  CUDADirichletBC& operator=(CUDADirichletBC&& bc)
  {
    _num_dofs = bc._num_dofs;
    _num_owned_boundary_dofs = bc._num_owned_boundary_dofs;
    _num_boundary_dofs = bc._num_boundary_dofs;
    _ddof_markers = bc._ddof_markers;
    _ddof_indices = bc._ddof_indices;
    _ddof_values = bc._ddof_values;
    bc._num_dofs = 0;
    bc._num_owned_boundary_dofs = 0;
    bc._num_boundary_dofs = 0;
    bc._ddof_markers = 0;
    bc._ddof_indices = 0;
    bc._ddof_values = 0;
    return *this;
  }
  //-----------------------------------------------------------------------------

  /// Update device-side values for all provided boundary conditions
  /// The user is responsible for ensuring the provided conditions are in the original list
  void update(const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T,U>>>& bcs) {
    for (auto const& bc: bcs) {
      bc->set(std::span<T>(_dof_values), {});
    }

    CUDA::safeMemcpyHtoD(_ddof_values, _dof_values.data(), _num_dofs * sizeof(T));
  }

  /// Get the number of degrees of freedom
  int32_t num_dofs() const { return _num_dofs; }

  /// Get a handle to the device-side dof markers
  CUdeviceptr dof_markers() const { return _ddof_markers; }

  /// Get the number of owned degrees of freedom subject to boundary
  /// conditions 
  int32_t num_owned_boundary_dofs() const { return _num_owned_boundary_dofs; }
  
  /// Get the number of degrees of freedom subject to boundary
  /// conditions 
  int32_t num_boundary_dofs() const { return _num_boundary_dofs; }

  /// Get a handle to the device-side dof indices
  CUdeviceptr dof_indices() const { return _ddof_indices; }

  /// Get a handle to the device-side dofs for the values
  CUdeviceptr dof_value_indices() const { return _ddof_indices; }

  /// Get a handle to the device-side dof values
  CUdeviceptr dof_values() const { return _ddof_values; }

private:
  /// The number of degrees of freedom
  int32_t _num_dofs;

  /// The number of degrees of freedom owned by the current process
  /// that are subject to the essential boundary conditions.
  int32_t _num_owned_boundary_dofs;

  /// The number of degrees of freedom that are subject to the
  /// essential boundary conditions, including ghost nodes.
  int32_t _num_boundary_dofs;

  /// A host-side vector with the values for the boundary conditions
  /// Used for cases when the boundary condition values change
  std::vector<T> _dof_values;

  /// Markers for each degree of freedom, indicating whether or not
  /// they are subject to essential boundary conditions
  CUdeviceptr _ddof_markers;

  /// Indices of the degrees of freedom that are subject to essential
  /// boundary conditions
  CUdeviceptr _ddof_indices;

  /// Values for each degree of freedom, indicating whether or not
  /// they are subject to essential boundary conditions
  CUdeviceptr _ddof_values;
};

} // namespace fem
} // namespace dolfinx

