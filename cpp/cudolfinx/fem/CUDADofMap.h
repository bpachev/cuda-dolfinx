// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of cuDOLFINX
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cudolfinx/common/CUDA.h>
#include <cuda.h>
#include <map>

namespace dolfinx {
namespace fem {
class DofMap;

/// A wrapper for a cellwise-to-global mapping of degres of freedom
/// that is stored in the device memory of a CUDA device.
class CUDADofMap
{
public:
  /// Create an empty dofmap
  CUDADofMap();

  /// Create a dofmap
  ///
  /// @param[in] dofmap The dofmap to copy to device memory
  CUDADofMap(
    const dolfinx::fem::DofMap& dofmap,
    std::int32_t offset,
    std::int32_t ghost_offset,
    std::map<std::int32_t, std::int32_t>* restriction
  );

  // constructors without restriction
  CUDADofMap(const dolfinx::fem::DofMap* dofmap);

  CUDADofMap(const dolfinx::fem::DofMap& dofmap);
   
  /// Alternate constructor
  CUDADofMap(
    const dolfinx::fem::DofMap* dofmap,
    std::int32_t offset,
    std::int32_t ghost_offset,
    std::map<std::int32_t, std::int32_t>* restriction
  );

  /// Destructor
  ~CUDADofMap();

  /// Copy constructor
  /// @param[in] dofmap The object to be copied
  CUDADofMap(const CUDADofMap& dofmap) = delete;

  /// Move constructor
  /// @param[in] dofmap The object to be moved
  CUDADofMap(CUDADofMap&& dofmap);

  /// Assignment operator
  /// @param[in] dofmap Another CUDADofMap object
  CUDADofMap& operator=(const CUDADofMap& dofmap) = delete;

  /// Move assignment operator
  /// @param[in] dofmap Another CUDADofMap object
  CUDADofMap& operator=(CUDADofMap&& dofmap);

  /// Update the dofmap on the device, possibly with a new restriction
  void update(std::int32_t offset, std::int32_t ghost_offset, std::map<std::int32_t, std::int32_t>* restriction);

  /// Get the underlying dofmap on the host
  const dolfinx::fem::DofMap* dofmap() const { return _dofmap; }

  /// Get the number of degrees of freedom
  int32_t num_dofs() const { return _num_dofs; }

  /// Get the number of cells
  int32_t num_cells() const { return _num_cells; }

  /// Get the number of dofs per cell
  int32_t num_dofs_per_cell() const {
    return _num_dofs_per_cell; }

  /// Get a handle to the device-side dofs of each cell
  CUdeviceptr dofs_per_cell() const {
    return _ddofs_per_cell; }

  /// Get the offsets to the first cell containing each degree of freedom
  CUdeviceptr cells_per_dof_ptr() const {
    return _dcells_per_dof_ptr; }

  /// Get the cells containing each degree of freedom
  CUdeviceptr cells_per_dof() const {
    return _dcells_per_dof; }

private:
  /// The underlying dofmap on the host
  const dolfinx::fem::DofMap* _dofmap;

  /// The number of degrees of freedom
  int32_t _num_dofs;

  /// The number of cells in the mesh
  int32_t _num_cells;

  /// The number of degrees of freedom in each cell
  int32_t _num_dofs_per_cell;

  /// The block size
  int32_t _block_size;

  /// The degrees of freedom of each cell
  CUdeviceptr _ddofs_per_cell;

  /// Offsets to the first cell containing each degree of freedom
  CUdeviceptr _dcells_per_dof_ptr;

  /// The cells containing each degree of freedom
  CUdeviceptr _dcells_per_dof;
};

} // namespace fem
} // namespace dolfinx

