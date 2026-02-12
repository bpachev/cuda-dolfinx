// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of cuDOLFINX
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <cudolfinx/common/CUDA.h>
#include <cudolfinx/fem/CUDADofMap.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/interpolate.h>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>
#include <cudolfinx/fem/CUDAInterpolate.h>
#include <basix/interpolation.h>

namespace dolfinx::fem
{
/// @brief a wrapper around a Function
template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_t<T>>
class CUDACoefficient
{
public:
  
  /// @brief Construct a new CUDACoefficient
  CUDACoefficient(std::shared_ptr<const Function<T, U>> f)
  {
    _f = f;
    _x = f->x();
    _values.assign(_x->array().begin(), _x->array().end());
    _dvalues_size = _x->bs() * (_x->index_map()->size_local()+_x->index_map()->num_ghosts()) * sizeof(T);
    CUDA::safeMemAlloc(&_dvalues, _dvalues_size);
    copy_host_values_to_device();

    // Count total no. of cells
    auto mesh = f->function_space()->mesh();
    auto map = mesh->topology()->index_map(mesh->topology()->dim());
    _num_cells = map->size_local() + map->num_ghosts();

    const int bs = _f->function_space()->dofmap()->bs();
    // Create global-to-cell DOF map used for interpolation
    _M = CUDA::create_interpolation_map(*_f);
    CUDA::safeMemAlloc(&_dM, _M.size()*sizeof(int));
    CUDA::safeMemcpyHtoD(_dM, (void*)(_M.data()), _M.size()*sizeof(int));
  }

  /// Copy to device, allocating GPU memory if required
  void copy_host_values_to_device()
  {
    CUDA::safeMemcpyHtoD(_dvalues, (void*)(_x->array().data()), _dvalues_size);
  }

  void copy_device_values_to_host()
  {
    CUDA::safeMemcpyDtoH((void*)_values.data(), _dvalues, _dvalues_size);
  }

  /// @brief Interpolate from CUDACoefficient `d_g` associated with the same mesh over all cells.
  /// This updates both host and device coefficient vectors of this object (not the host-side Function).
  ///
  /// @pre Both functions must share the same reference map.
  void interpolate(const CUDACoefficient<T, U>& d_g)
  {
    auto element0 = d_g.host_function()->function_space()->element();
    assert(element0);
    auto element1 = _f->function_space()->element();

    if (element0->map_type() != element1->map_type()) {
      throw std::runtime_error("Functions must share the same reference element mapping.");
    }

    if (d_g.host_function()->function_space()->dofmap()->bs() !=
        _f->function_space()->dofmap()->bs()) {
      throw std::runtime_error("Functions must have the same block size.");
    }

    // Device-side cofficient vector of g
    CUdeviceptr _dvalues_g = d_g.device_values();

    // Create interpolation operator IM, mapping g to f
    auto [IM, _im_shape] = basix::compute_interpolation_operator<T>(element0->basix_element(), element1->basix_element());
    CUdeviceptr dIM;
    CUDA::safeMemAlloc(&dIM, IM.size()*sizeof(T));
    CUDA::safeMemcpyHtoD(dIM, (void*)(IM.data()), IM.size()*sizeof(T));


    const int bs = _f->function_space()->dofmap()->bs();
    CUDA::interpolate_same_map<T>(_dvalues, _dvalues_g, _im_shape, _num_cells, bs, dIM, _dM, d_g.device_dof_matrix());

    copy_device_values_to_host();
    cuMemFree(dIM);
  }


  /// Return a copy of host-side coefficient vector
  std::vector<T> values() const
  {
    return _values;
  }

  /// Get pointer to vector data on device
  CUdeviceptr device_values() const
  {
    return _dvalues;
  }

  /// Get pointer to the underlying Function
  std::shared_ptr<const dolfinx::fem::Function<T,U>> host_function() const
  {
    return _f;
  }

  CUdeviceptr device_dof_matrix() const
  {
    return _dM;
  }

  ~CUDACoefficient() {
    if (_dvalues)
      cuMemFree(_dvalues);
    if (_dM)
        cuMemFree(_dM);
  }

private:
  // Host-side coefficient array. Any time _dvalues is updated, this is also updated.
  std::vector<T> _values;
  // Device-side coefficient array
  CUdeviceptr _dvalues;
  // Size of coefficient array
  size_t _dvalues_size;
  // Pointer to host-side Function
  std::shared_ptr<const dolfinx::fem::Function<T, U>> _f;
  // Pointer to host-side coefficient vector
  std::shared_ptr<const dolfinx::la::Vector<T>> _x;

  // Total number of cells
  size_t _num_cells;

  // Interpolation maps
  std::vector<std::int32_t> _M;
  CUdeviceptr _dM;
};

template class dolfinx::fem::CUDACoefficient<double>;
}
