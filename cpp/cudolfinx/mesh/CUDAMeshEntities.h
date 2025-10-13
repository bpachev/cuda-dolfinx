// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of cuDOLFINX
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cudolfinx/common/CUDA.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <cuda.h>

namespace dolfinx {
namespace mesh {

/// A wrapper for data related to mesh entities of a given dimension,
/// stored in the device memory of a CUDA device.
template <std::floating_point T>
class CUDAMeshEntities
{
public:

  //-----------------------------------------------------------------------------
  /// Create an empty set of mesh entities
  CUDAMeshEntities()
    : _tdim()
    , _dim()
    , _num_cells()
    , _num_mesh_entities()
    , _num_mesh_entities_per_cell()
    , _dmesh_entities_per_cell(0)
    , _dcells_per_mesh_entity_ptr(0)
    , _dcells_per_mesh_entity(0)
    , _dmesh_entity_permutations(0)
  {
  }
  //-----------------------------------------------------------------------------
  /// Create a set of mesh entities from a mesh
  ///
  /// @param[in] mesh Data structures for mesh topology and geometry
  /// @param[in] dim The dimension of mesh entities
  CUDAMeshEntities(
    const dolfinx::mesh::Mesh<T>& mesh,
    int dim)
    : _tdim(mesh.topology()->dim())
    , _dim(dim)
    , _num_cells(mesh.geometry().dofmap().extent(0))
  {
    CUresult cuda_err;
    const char * cuda_err_description;

    // Get the data related to mesh entities
    mesh.topology_mutable()->create_entities(_dim);
    mesh.topology_mutable()->create_connectivity(_dim, _tdim);
    mesh.topology_mutable()->create_entity_permutations();
  const graph::AdjacencyList<std::int32_t>& cells_to_mesh_entities =
      *mesh.topology()->connectivity(_tdim, _dim);

    // Allocate device-side storage for mesh entities of each cell
    assert(_num_cells == cells_to_mesh_entities.num_nodes());
    _num_mesh_entities_per_cell = cells_to_mesh_entities.num_links(0);
    if (_num_cells > 0 && _num_mesh_entities_per_cell > 0) {
      size_t dmesh_entities_per_cell_size =
        _num_cells * _num_mesh_entities_per_cell * sizeof(int32_t);
      cuda_err = cuMemAlloc(
        &_dmesh_entities_per_cell,
        dmesh_entities_per_cell_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemAlloc() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }

      // Copy mesh entities of each cell to device
      const std::vector<int32_t>&
        mesh_entities_per_cell = cells_to_mesh_entities.array();
      cuda_err = cuMemcpyHtoD(
        _dmesh_entities_per_cell,
        mesh_entities_per_cell.data(),
        dmesh_entities_per_cell_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
    }

    // Allocate device-side storage for offsets to the cells of each
    // mesh entity
    const graph::AdjacencyList<std::int32_t>& mesh_entities_to_cells =
      *mesh.topology()->connectivity(_dim, _tdim);
    const std::vector<std::int32_t>&
      cells_per_mesh_entity_ptr = mesh_entities_to_cells.offsets();
    _num_mesh_entities = mesh_entities_to_cells.num_nodes();
    assert(_num_mesh_entities+1 == cells_per_mesh_entity_ptr.size());
    if (_num_mesh_entities >= 0) {
      size_t dcells_per_mesh_entity_ptr_size =
        (_num_mesh_entities+1) * sizeof(int32_t);
      cuda_err = cuMemAlloc(
        &_dcells_per_mesh_entity_ptr,
        dcells_per_mesh_entity_ptr_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemAlloc() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
      cuda_err = cuMemcpyHtoD(
        _dcells_per_mesh_entity_ptr,
        cells_per_mesh_entity_ptr.data(),
        dcells_per_mesh_entity_ptr_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
    }

    // Allocate device-side storage for the cells of each mesh entity
    const std::vector<std::int32_t>&
      cells_per_mesh_entity = mesh_entities_to_cells.array();
    assert(_num_mesh_entities < 0 ||
           cells_per_mesh_entity_ptr[_num_mesh_entities] == cells_per_mesh_entity.size());
    assert(_num_cells * _num_mesh_entities_per_cell == cells_per_mesh_entity.size());
    if (_num_mesh_entities >= 0 && cells_per_mesh_entity_ptr[_num_mesh_entities] > 0) {
      size_t dcells_per_mesh_entity_size =
        cells_per_mesh_entity_ptr[_num_mesh_entities] * sizeof(int32_t);
      cuda_err = cuMemAlloc(
        &_dcells_per_mesh_entity,
        dcells_per_mesh_entity_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemAlloc() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
      cuda_err = cuMemcpyHtoD(
        _dcells_per_mesh_entity,
        cells_per_mesh_entity.data(),
        dcells_per_mesh_entity_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
    }

    if (_dim == _tdim-1) {
      // Obtain cell permutations
      mesh.topology_mutable()->create_entity_permutations();
      auto mesh_entity_permutations = mesh.topology()->get_facet_permutations();

      // Allocate device-side storage for mesh entity permutations
      if (_num_mesh_entities_per_cell > 0 && _num_cells > 0) {
        size_t dmesh_entity_permutations_size =
          _num_mesh_entities_per_cell * _num_cells * sizeof(uint8_t);
        cuda_err = cuMemAlloc(
          &_dmesh_entity_permutations,
          dmesh_entity_permutations_size);
        if (cuda_err != CUDA_SUCCESS) {
          cuGetErrorString(cuda_err, &cuda_err_description);
          throw std::runtime_error(
            "cuMemAlloc() failed with " + std::string(cuda_err_description) +
            " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
        }
        cuda_err = cuMemcpyHtoD(
          _dmesh_entity_permutations,
          mesh_entity_permutations.data(),
          dmesh_entity_permutations_size);
        if (cuda_err != CUDA_SUCCESS) {
          cuMemFree(_dmesh_entity_permutations);
          cuGetErrorString(cuda_err, &cuda_err_description);
          throw std::runtime_error(
            "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
            " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
        }
      }
    } else {
      _dmesh_entity_permutations = 0;
    }
  }
  //-----------------------------------------------------------------------------
  /// Destructor
  ~CUDAMeshEntities()
  {
    if (_dmesh_entity_permutations)
      cuMemFree(_dmesh_entity_permutations);
    if (_dcells_per_mesh_entity)
      cuMemFree(_dcells_per_mesh_entity);
    if (_dcells_per_mesh_entity_ptr)
      cuMemFree(_dcells_per_mesh_entity_ptr);
    if (_dmesh_entities_per_cell)
      cuMemFree(_dmesh_entities_per_cell);
  }
  //-----------------------------------------------------------------------------
  /// Copy constructor
  /// @param[in] mesh_entities The object to be copied
  CUDAMeshEntities(const CUDAMeshEntities& mesh_entities) = delete;

  /// Move constructor
  /// @param[in] mesh_entities The object to be moved
  CUDAMeshEntities(CUDAMeshEntities&& mesh)
    : _tdim(mesh._tdim)
    , _dim(mesh._dim)
    , _num_cells(mesh._num_cells)
    , _num_mesh_entities(mesh._num_mesh_entities)
    , _num_mesh_entities_per_cell(mesh._num_mesh_entities_per_cell)
    , _dmesh_entities_per_cell(mesh._dmesh_entities_per_cell)
    , _dcells_per_mesh_entity_ptr(mesh._dcells_per_mesh_entity_ptr)
    , _dcells_per_mesh_entity(mesh._dcells_per_mesh_entity)
    , _dmesh_entity_permutations(mesh._dmesh_entity_permutations)
  {
    mesh._tdim = 0;
    mesh._dim = 0;
    mesh._num_cells = 0;
    mesh._num_mesh_entities = 0;
    mesh._num_mesh_entities_per_cell = 0;
    mesh._dmesh_entities_per_cell = 0;
    mesh._dcells_per_mesh_entity_ptr = 0;
    mesh._dcells_per_mesh_entity = 0;
    mesh._dmesh_entity_permutations = 0;
  }
  //-----------------------------------------------------------------------------
  /// Assignment operator
  /// @param[in] mesh_entities Another CUDAMeshEntities object
  CUDAMeshEntities& operator=(const CUDAMeshEntities& mesh_entities) = delete;

  /// Move assignment operator
  /// @param[in] mesh_entities Another CUDAMeshEntities object
  CUDAMeshEntities& operator=(CUDAMeshEntities&& mesh)
  {
    _tdim = mesh._tdim;
    _dim = mesh._dim;
    _num_cells = mesh._num_cells;
    _num_mesh_entities = mesh._num_mesh_entities;
    _num_mesh_entities_per_cell = mesh._num_mesh_entities_per_cell;
    _dmesh_entities_per_cell = mesh._dmesh_entities_per_cell;
    _dcells_per_mesh_entity_ptr = mesh._dcells_per_mesh_entity_ptr;
    _dcells_per_mesh_entity = mesh._dcells_per_mesh_entity;
    _dmesh_entity_permutations = mesh._dmesh_entity_permutations;
    mesh._tdim = 0;
    mesh._dim = 0;
    mesh._num_cells = 0;
    mesh._num_mesh_entities = 0;
    mesh._num_mesh_entities_per_cell = 0;
    mesh._dmesh_entities_per_cell = 0;
    mesh._dcells_per_mesh_entity_ptr = 0;
    mesh._dcells_per_mesh_entity = 0;
    mesh._dmesh_entity_permutations = 0;
    return *this;
  }
  //-----------------------------------------------------------------------------

  /// Get the topological dimension of the mesh
  int32_t tdim() const { return _tdim; }

  /// Get the dimension of the mesh entities
  int32_t dim() const { return _dim; }

  /// Get the number of cells in the mesh
  int32_t num_cells() const { return _num_cells; }

  /// Get the number of mesh entities
  int32_t num_mesh_entities() const { return _num_mesh_entities; }

  /// Get the number of mesh entities in a cell
  int32_t num_mesh_entities_per_cell() const {
    return _num_mesh_entities_per_cell; }

  /// Get a handle to the device-side mesh entities of each cell
  CUdeviceptr mesh_entities_per_cell() const {
    return _dmesh_entities_per_cell; }

  /// Get a handle to the device-side offsets to the mesh cells of
  /// each mesh entity
  CUdeviceptr cells_per_mesh_entity_ptr() const {
    return _dcells_per_mesh_entity_ptr; }

  /// Get a handle to the device-side mesh cells of each mesh entity
  CUdeviceptr cells_per_mesh_entity() const {
    return _dcells_per_mesh_entity; }

  /// Get a handle to the device-side mesh entity permutations
  CUdeviceptr mesh_entity_permutations() const {
    return _dmesh_entity_permutations; }

private:
  /// The topological dimension of the mesh, or the largest dimension
  /// of any of the mesh entities
  int32_t _tdim;

  /// The dimension of the mesh entities
  int32_t _dim;

  /// The number of cells in the mesh
  int32_t _num_cells;

  /// The number of mesh entities
  int32_t _num_mesh_entities;

  /// The number of mesh entities in a cell
  int32_t _num_mesh_entities_per_cell;

  /// The mesh entities of each cell
  CUdeviceptr _dmesh_entities_per_cell;

  /// Offsets to the first cell containing each mesh entity
  CUdeviceptr _dcells_per_mesh_entity_ptr;

  /// The cells containing each mesh entity
  CUdeviceptr _dcells_per_mesh_entity;

  /// Mesh entity permutations
  CUdeviceptr _dmesh_entity_permutations;
};

} // namespace mesh
} // namespace dolfinx

