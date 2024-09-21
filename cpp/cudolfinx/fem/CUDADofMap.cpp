// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CUDADofMap.h"
#include <cudolfinx/common/CUDA.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <cuda.h>

using namespace dolfinx;
using namespace dolfinx::fem;

//-----------------------------------------------------------------------------
CUDADofMap::CUDADofMap()
  : _dofmap(nullptr)
  , _num_dofs()
  , _num_cells()
  , _num_dofs_per_cell()
  , _ddofs_per_cell(0)
  , _dcells_per_dof_ptr(0)
  , _dcells_per_dof(0)
{
}

CUDADofMap::CUDADofMap(
  const dolfinx::fem::DofMap* dofmap)
  : CUDADofMap::CUDADofMap(*dofmap)
{
}

//-----------------------------------------------------------------------------
CUDADofMap::CUDADofMap(
  const dolfinx::fem::DofMap& dofmap)
  : _dofmap(&dofmap)
  , _num_dofs()
  , _num_cells()
  , _num_dofs_per_cell()
  , _ddofs_per_cell(0)
  , _dcells_per_dof_ptr(0)
  , _dcells_per_dof(0)
{
  CUresult cuda_err;
  const char * cuda_err_description;

  auto dofs = dofmap.map();
  auto element_dof_layout = dofmap.element_dof_layout();
  // get block sizes and ensure positivity (sometimes the default is -1)
  std::int32_t element_block_size = element_dof_layout.block_size();
  std::int32_t block_size = dofmap.bs();
  element_block_size = (element_block_size < 0) ? 1 : element_block_size;
  block_size = (block_size < 0) ? 1 : block_size;
  _num_cells = dofs.extent(0);
  _num_dofs_per_cell = element_dof_layout.num_dofs() * element_block_size;
  _num_dofs = dofs.size() * block_size;
  std::vector<std::int32_t> unrolled_dofs;
  const std::int32_t* dofs_per_cell;

  if (_num_dofs != _num_cells * _num_dofs_per_cell) {
    throw std::runtime_error(
       "Num dofs " + std::to_string(_num_dofs) + " != " + std::to_string(_num_cells) +
       "*" + std::to_string(_num_dofs_per_cell) + " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
  if (block_size == 1) {
    dofs_per_cell = dofs.data_handle();
  }
  else {
    unrolled_dofs.resize(_num_dofs);
    const std::int32_t* dofs_1d = dofs.data_handle();
    for (std::size_t i = 0; i < _num_dofs; i++)
      unrolled_dofs[i] = block_size*dofs_1d[i/block_size] + i%block_size;

    dofs_per_cell = unrolled_dofs.data();
  }

  // Allocate device-side storage for degrees of freedom
  if (_num_cells > 0 && _num_dofs_per_cell > 0) {
    size_t ddofs_per_cell_size = _num_dofs * sizeof(int32_t);
    cuda_err = cuMemAlloc(
      &_ddofs_per_cell,
      ddofs_per_cell_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Copy cell degrees of freedom to device
    cuda_err = cuMemcpyHtoD(
      _ddofs_per_cell, dofs_per_cell, ddofs_per_cell_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuMemFree(_ddofs_per_cell);
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }
  // cells_per_dof_ptr and cells_per_dof are only used for
  // lookup table computations, which currently aren't in use
/*
  // Compute mapping from degrees of freedom to cells
  std::vector<int32_t> cells_per_dof_ptr(_num_dofs+1);

  // Count the number cells containing each degree of freedom
  for (int32_t i = 0; i < _num_cells; i++) {
    auto cell_dofs = dofmap.cell_dofs(i);
    for (int32_t l = 0; l < cell_dofs.size(); l++) {
      int32_t j = cell_dofs[l];
      cells_per_dof_ptr[j+1]++;
    }
  }

  // Compute offset to the first cell for each degree of freedom
  for (int32_t i = 0; i < _num_dofs; i++)
    cells_per_dof_ptr[i+1] += cells_per_dof_ptr[i];
  int32_t num_dof_cells = cells_per_dof_ptr[_num_dofs];
  if (num_dof_cells != _num_cells * _num_dofs_per_cell) {
    cuMemFree(_ddofs_per_cell);
    throw std::logic_error(
      "Expected " + std::to_string(_num_cells) + " cells, " +
      std::to_string(_num_dofs_per_cell) + " degrees of freedom per cell, "
      "but the mapping from degrees of freedom to cells contains " +
      std::to_string(num_dof_cells) + " values" );
  }

  // Allocate storage for and compute the cells containing each degree
  // of freedom
  std::vector<int32_t> cells_per_dof(num_dof_cells);
  for (int32_t i = 0; i < _num_cells; i++) {
    auto cell_dofs = dofmap.cell_dofs(i);
    for (int32_t l = 0; l < cell_dofs.size(); l++) {
      int32_t j = cell_dofs[l];
      int32_t p = cells_per_dof_ptr[j];
      cells_per_dof[p] = i;
      cells_per_dof_ptr[j]++;
    }
  }

  // Adjust offsets to first cell
  for (int32_t i = _num_dofs; i > 0; i--)
    cells_per_dof_ptr[i] = cells_per_dof_ptr[i-1];
  cells_per_dof_ptr[0] = 0;

  // Allocate device-side storage for offsets to the first cell
  // containing each degree of freedom
  if (_num_dofs > 0) {
    size_t dcells_per_dof_ptr_size = (_num_dofs+1) * sizeof(int32_t);
    cuda_err = cuMemAlloc(
      &_dcells_per_dof_ptr, dcells_per_dof_ptr_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      cuMemFree(_ddofs_per_cell);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Copy cell degrees of freedom to device
    cuda_err = cuMemcpyHtoD(
      _dcells_per_dof_ptr, cells_per_dof_ptr.data(), dcells_per_dof_ptr_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      cuMemFree(_dcells_per_dof_ptr);
      cuMemFree(_ddofs_per_cell);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }

  // Allocate device-side storage for cells containing each degree of freedom
  if (_num_cells > 0 && _num_dofs_per_cell > 0) {
    size_t dcells_per_dof_size = num_dof_cells * sizeof(int32_t);
    cuda_err = cuMemAlloc(
      &_dcells_per_dof,
      dcells_per_dof_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      cuMemFree(_dcells_per_dof_ptr);
      cuMemFree(_ddofs_per_cell);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Copy cell degrees of freedom to device
    cuda_err = cuMemcpyHtoD(
      _dcells_per_dof, cells_per_dof.data(), dcells_per_dof_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      cuMemFree(_dcells_per_dof);
      cuMemFree(_dcells_per_dof_ptr);
      cuMemFree(_ddofs_per_cell);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }*/
}
//-----------------------------------------------------------------------------
CUDADofMap::~CUDADofMap()
{
  if (_dcells_per_dof)
    cuMemFree(_dcells_per_dof);
  if (_dcells_per_dof_ptr)
    cuMemFree(_dcells_per_dof_ptr);
  if (_ddofs_per_cell)
    cuMemFree(_ddofs_per_cell);
}
//-----------------------------------------------------------------------------
CUDADofMap::CUDADofMap(CUDADofMap&& dofmap)
  : _dofmap(dofmap._dofmap)
  , _num_dofs(dofmap._num_dofs)
  , _num_cells(dofmap._num_cells)
  , _num_dofs_per_cell(dofmap._num_dofs_per_cell)
  , _ddofs_per_cell(dofmap._ddofs_per_cell)
  , _dcells_per_dof_ptr(dofmap._dcells_per_dof_ptr)
  , _dcells_per_dof(dofmap._dcells_per_dof)
{
  dofmap._dofmap = nullptr;
  dofmap._num_dofs = 0;
  dofmap._num_cells = 0;
  dofmap._num_dofs_per_cell = 0;
  dofmap._ddofs_per_cell = 0;
  dofmap._dcells_per_dof_ptr = 0;
  dofmap._dcells_per_dof = 0;
}
//-----------------------------------------------------------------------------
CUDADofMap& CUDADofMap::operator=(CUDADofMap&& dofmap)
{
  _dofmap = dofmap._dofmap;
  _num_dofs = dofmap._num_dofs;
  _num_cells = dofmap._num_cells;
  _num_dofs_per_cell = dofmap._num_dofs_per_cell;
  _ddofs_per_cell = dofmap._ddofs_per_cell;
  _dcells_per_dof_ptr = dofmap._dcells_per_dof_ptr;
  _dcells_per_dof = dofmap._dcells_per_dof;
  dofmap._dofmap = nullptr;
  dofmap._num_dofs = 0;
  dofmap._num_cells = 0;
  dofmap._num_dofs_per_cell = 0;
  dofmap._ddofs_per_cell = 0;
  dofmap._dcells_per_dof_ptr = 0;
  dofmap._dcells_per_dof = 0;
  return *this;
}
//-----------------------------------------------------------------------------
