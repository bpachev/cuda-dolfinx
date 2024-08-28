// Copyright (C) 2020 James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/fem/Form.h>

#if defined(HAS_CUDA_TOOLKIT)
#include <cudolfinx/common/CUDA.h>
#include <cudolfinx/common/CUDAStore.h>
#include <cudolfinx/fem/CUDACoefficient.h>
#include <cudolfinx/fem/CUDADofMap.h>
#include <cudolfinx/la/CUDAVector.h>
#endif
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/common/IndexMap.h>

#if defined(HAS_CUDA_TOOLKIT)
#include <cuda.h>
#endif

#include <petscvec.h>

#if defined(HAS_CUDA_TOOLKIT)
namespace dolfinx {
namespace fem {

/// A wrapper for a form coefficient with data that is stored in the
/// device memory of a CUDA device.
template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
class CUDAFormCoefficients
{
public:
  /// Scalar Type
  using scalar_type = T;

  /// Create an empty collection coefficient values
  //-----------------------------------------------------------------------------
  CUDAFormCoefficients()
    : _coefficients(nullptr)
    , _dofmaps_num_dofs_per_cell(0)
    , _dofmaps_dofs_per_cell(0)
    , _coefficient_values_offsets(0)
    , _coefficient_values(0)
    , _num_cells()
    , _num_packed_coefficient_values_per_cell()
    , _page_lock(false)
    , _host_coefficient_values()
    , _dpacked_coefficient_values(0)
  {
  }

  /// Create a collection coefficient values from a given form
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] form The variational form whose coefficients are used
  /// @param[in] dofmap_store A cache mapping host-side to device-side dofmaps
  /// @param[in] page_lock Whether or not to use page-locked memory
  ///                      for host-side arrays
  //-----------------------------------------------------------------------------
  CUDAFormCoefficients(
    const CUDA::Context& cuda_context,
    Form<T,U>* form,
    common::CUDAStore<DofMap, CUDADofMap>& dofmap_store, 
    bool page_lock=false)
    : _coefficients(form->coefficients())
    , _dofmaps_num_dofs_per_cell(0)
    , _dofmaps_dofs_per_cell(0)
    , _coefficient_values_offsets(0)
    , _coefficient_values(0)
    , _num_cells()
    , _num_packed_coefficient_values_per_cell()
    , _page_lock(page_lock)
    , _host_coefficient_values()
    , _dpacked_coefficient_values(0)
  {
    CUresult cuda_err;
    const char * cuda_err_description;
    int num_coefficients = _coefficients.size();
    std::vector<int> offsets = {0};
    for (const auto & c : _coefficients)
    {
      if (!c)
        throw std::runtime_error("Not all form coefficients have been set.");
      offsets.push_back(offsets.back() + c->function_space()->element()->space_dimension());
    }

    // Get the number of cells in the mesh
    std::shared_ptr<const mesh::Mesh<U>> mesh = form->mesh();
    const int tdim = mesh->topology()->dim();
    _num_cells = mesh->topology()->index_map(tdim)->size_local()
      + mesh->topology()->index_map(tdim)->num_ghosts();
    _num_packed_coefficient_values_per_cell = offsets.back();

    // Allocate device-side storage for number of dofs per cell and
    // pointers to the dofs per cell for each coefficient
    if (num_coefficients > 0) {
      std::vector<int> dofmaps_num_dofs_per_cell(num_coefficients);
      std::vector<CUdeviceptr> dofmaps_dofs_per_cell(num_coefficients);
      for (int i = 0; i < num_coefficients; i++) {
        _device_coefficients.push_back(std::make_shared<CUDACoefficient<T,U>>(_coefficients[i]));
	_coefficient_device_ptrs.push_back(_device_coefficients[i]->device_values());
        const fem::CUDADofMap* cuda_dofmap =
          dofmap_store.get_device_object(_coefficients[i]->function_space()->dofmap()).get();
        dofmaps_num_dofs_per_cell[i] = cuda_dofmap->num_dofs_per_cell();
        dofmaps_dofs_per_cell[i] = cuda_dofmap->dofs_per_cell();
      }

      size_t dofmaps_num_dofs_per_cell_size = num_coefficients * sizeof(int);
      cuda_err = cuMemAlloc(
        &_dofmaps_num_dofs_per_cell, dofmaps_num_dofs_per_cell_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemAlloc() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
      cuda_err = cuMemcpyHtoD(
        _dofmaps_num_dofs_per_cell, dofmaps_num_dofs_per_cell.data(),
        dofmaps_num_dofs_per_cell_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        cuMemFree(_dofmaps_num_dofs_per_cell);
        throw std::runtime_error(
          "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }

      size_t dofmaps_dofs_per_cell_size = num_coefficients * sizeof(CUdeviceptr);
      cuda_err = cuMemAlloc(
        &_dofmaps_dofs_per_cell, dofmaps_dofs_per_cell_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        cuMemFree(_dofmaps_num_dofs_per_cell);
        throw std::runtime_error(
          "cuMemAlloc() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
      cuda_err = cuMemcpyHtoD(
        _dofmaps_dofs_per_cell, dofmaps_dofs_per_cell.data(), dofmaps_dofs_per_cell_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        cuMemFree(_dofmaps_dofs_per_cell);
        cuMemFree(_dofmaps_num_dofs_per_cell);
        throw std::runtime_error(
          "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
    }

    // Allocate device-side storage for offsets to and pointers to
    // coefficient values and copy to device
    if (num_coefficients > 0) {
      size_t coefficient_values_offsets_size = offsets.size() * sizeof(int);
      cuda_err = cuMemAlloc(
        &_coefficient_values_offsets, coefficient_values_offsets_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        cuMemFree(_dofmaps_dofs_per_cell);
        cuMemFree(_dofmaps_num_dofs_per_cell);
        throw std::runtime_error(
          "cuMemAlloc() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
      cuda_err = cuMemcpyHtoD(
        _coefficient_values_offsets, offsets.data(), coefficient_values_offsets_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        cuMemFree(_coefficient_values_offsets);
        cuMemFree(_dofmaps_dofs_per_cell);
        cuMemFree(_dofmaps_num_dofs_per_cell);
        throw std::runtime_error(
          "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }

      size_t coefficient_values_size = num_coefficients * sizeof(CUdeviceptr);
      cuda_err = cuMemAlloc(
        &_coefficient_values, coefficient_values_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        cuMemFree(_coefficient_values_offsets);
        cuMemFree(_dofmaps_dofs_per_cell);
        cuMemFree(_dofmaps_num_dofs_per_cell);
        throw std::runtime_error(
          "cuMemAlloc() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }

      size_t coefficient_indices_size = num_coefficients * sizeof(int);
      CUDA::safeMemAlloc(&_coefficient_indices, coefficient_indices_size);
    }

    // Allocate device-side storage for packed coefficient values
    if (_num_cells > 0 && _num_packed_coefficient_values_per_cell > 0) {
      size_t dpacked_coefficient_values_size =
        _num_cells * _num_packed_coefficient_values_per_cell * sizeof(PetscScalar);
      cuda_err = cuMemAlloc(
        &_dpacked_coefficient_values, dpacked_coefficient_values_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        cuMemFree(_coefficient_values);
        cuMemFree(_coefficient_values_offsets);
        cuMemFree(_dofmaps_dofs_per_cell);
        cuMemFree(_dofmaps_num_dofs_per_cell);
        throw std::runtime_error(
          "cuMemAlloc() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
    }

  }

  /// Destructor
  ~CUDAFormCoefficients()
  {
    CUresult cuda_err;
    const char * cuda_err_description;
    if (_coefficient_values)
      cuMemFree(_coefficient_values);
    if (_coefficient_values_offsets)
      cuMemFree(_coefficient_values_offsets);
    if (_dofmaps_dofs_per_cell)
      cuMemFree(_dofmaps_dofs_per_cell);
    if (_dofmaps_num_dofs_per_cell)
      cuMemFree(_dofmaps_num_dofs_per_cell);
    if (_dpacked_coefficient_values)
      cuMemFree(_dpacked_coefficient_values);
  }

  /// Copy constructor
  /// @param[in] form_coefficient The object to be copied
  CUDAFormCoefficients(const CUDAFormCoefficients& form_coefficient) = delete;

  /// Assignment operator
  /// @param[in] form_coefficient Another CUDAFormCoefficients object
  CUDAFormCoefficients& operator=(const CUDAFormCoefficients& form_coefficient) = delete;


  /// Get the number of mesh cells that the coefficient applies to
  int32_t num_coefficients() const { return _coefficients.size(); }

  /// Return array of Functions
  const std::vector<std::shared_ptr<const Function<T, U>>>& coefficients() { return _coefficients; };

  /// Return array of CUDACoefficients (wrappers around Function)
  const std::vector<std::shared_ptr<CUDACoefficient<T,U>>>& device_coefficients() { return _device_coefficients; }

  /// Return array of device pointers for the coefficients
  const std::vector<CUdeviceptr> coefficient_device_ptrs() { return _coefficient_device_ptrs; }

  /// Get device-side pointer to number of dofs per cell for each coefficient
  CUdeviceptr dofmaps_num_dofs_per_cell() const { return _dofmaps_num_dofs_per_cell; }

  /// Get device-side pointer to dofs per cell for each coefficient
  CUdeviceptr dofmaps_dofs_per_cell() const { return _dofmaps_dofs_per_cell; }

  /// Get device-side pointer to offsets to coefficient values within
  /// a cell for each coefficient
  CUdeviceptr coefficient_values_offsets() const { return _coefficient_values_offsets; }

  /// Get device-side pointer to an array of pointers to coefficient
  /// values for each coefficient
  CUdeviceptr coefficient_values() const { return _coefficient_values; }

  /// Get device-side pointer to an array of coefficient indices
  /// This array is meant to be overwritten with each call to 
  /// pack_coefficients.
  CUdeviceptr coefficient_indices() const { return _coefficient_indices; }

  /// Get the number of mesh cells that the coefficient applies to
  int32_t num_cells() const { return _num_cells; }

  /// Get the number of coefficient values per cell
  int32_t num_packed_coefficient_values_per_cell() const {
      return _num_packed_coefficient_values_per_cell; }

  /// Get the coefficient values that the coefficient applies to
  CUdeviceptr packed_coefficient_values() const { return _dpacked_coefficient_values; }

  //-----------------------------------------------------------------------------
  /// Move constructor
  /// @param[in] form_coefficient The object to be moved
  CUDAFormCoefficients(
    CUDAFormCoefficients&& form_coefficients)
    : _coefficients(std::move(form_coefficients._coefficients))
    , _dofmaps_num_dofs_per_cell(form_coefficients._dofmaps_num_dofs_per_cell)
    , _dofmaps_dofs_per_cell(form_coefficients._dofmaps_dofs_per_cell)
    , _coefficient_values_offsets(form_coefficients._coefficient_values_offsets)
    , _coefficient_values(form_coefficients._coefficient_values)
    , _num_cells(form_coefficients._num_cells)
    , _num_packed_coefficient_values_per_cell(form_coefficients._num_packed_coefficient_values_per_cell)
    , _page_lock(form_coefficients._page_lock)
    , _dpacked_coefficient_values(form_coefficients._dpacked_coefficient_values)
  {
    form_coefficients._dofmaps_num_dofs_per_cell = 0;
    form_coefficients._dofmaps_dofs_per_cell = 0;
    form_coefficients._coefficient_values_offsets = 0;
    form_coefficients._coefficient_values = 0;
    form_coefficients._num_cells = 0;
    form_coefficients._num_packed_coefficient_values_per_cell = 0;
    form_coefficients._page_lock = false;
    std::swap(_host_coefficient_values, form_coefficients._host_coefficient_values);
    form_coefficients._dpacked_coefficient_values = 0;
  }
  //-----------------------------------------------------------------------------
  /// Move assignment operator
  /// @param[in] form_coefficient Another CUDAFormCoefficients object
  CUDAFormCoefficients& operator=(
    CUDAFormCoefficients&& form_coefficients)
  {
    _coefficients = std::move(form_coefficients._coefficients);
    _dofmaps_num_dofs_per_cell = form_coefficients._dofmaps_num_dofs_per_cell;
    _dofmaps_dofs_per_cell = form_coefficients._dofmaps_dofs_per_cell;
    _coefficient_values_offsets = form_coefficients._coefficient_values_offsets;
    _coefficient_values = form_coefficients._coefficient_values;
    _num_cells = form_coefficients._num_cells;
    _num_packed_coefficient_values_per_cell = form_coefficients._num_packed_coefficient_values_per_cell;
    _page_lock = form_coefficients._page_lock;
    std::swap(_host_coefficient_values, form_coefficients._host_coefficient_values);
    _dpacked_coefficient_values = form_coefficients._dpacked_coefficient_values;
    form_coefficients._dofmaps_num_dofs_per_cell = 0;
    form_coefficients._dofmaps_dofs_per_cell = 0;
    form_coefficients._coefficient_values_offsets = 0;
    form_coefficients._coefficient_values = 0;
    form_coefficients._num_cells = 0;
    form_coefficients._num_packed_coefficient_values_per_cell = 0;
    form_coefficients._page_lock = false;
    form_coefficients._dpacked_coefficient_values = 0;
    return *this;
  }
  //-----------------------------------------------------------------------------
  /// Update the coefficient values by copying values from host to device.
  /// This must be called before packing the coefficients, if they
  /// have been changed on the host.
  void copy_coefficients_to_device(
    const CUDA::Context& cuda_context)
  {
    for (int i = 0; i < _device_coefficients.size(); i++) {
      _device_coefficients[i]->copy_host_values_to_device();
    }
  }

private:
  /// The underlying coefficients on the host
  std::vector<std::shared_ptr<const Function<T, U>>> _coefficients;

  /// CUDA wrappers for the coefficients
  std::vector<std::shared_ptr<CUDACoefficient<T,U>>> _device_coefficients;

  /// Array of device pointers
  std::vector<CUdeviceptr> _coefficient_device_ptrs;

  /// Number of dofs per cell for each coefficient
  CUdeviceptr _dofmaps_num_dofs_per_cell;

  /// Dofs per cell for each coefficient
  CUdeviceptr _dofmaps_dofs_per_cell;

  /// Get device-side pointer to offsets to coefficient values within
  /// a cell for each coefficient
  CUdeviceptr _coefficient_values_offsets;

  /// Get device-side pointer to an array of pointers to coefficient
  /// values for each coefficient
  CUdeviceptr _coefficient_values;

  /// An array for storing coefficient indices for partial packing
  CUdeviceptr _coefficient_indices;

  /// The number of cells that the coefficient applies to
  int32_t _num_cells;

  /// The number of packed coefficient values per cell
  int32_t _num_packed_coefficient_values_per_cell;

  /// Whether or not the host-side array of values uses page-locked
  /// (pinned) memory
  bool _page_lock;

  /// Host-side array of coefficient values
  mutable std::vector<T> 
    _host_coefficient_values;

  /// The coefficient values
  CUdeviceptr _dpacked_coefficient_values;
};

} // namespace fem
} // namespace dolfinx
#endif
