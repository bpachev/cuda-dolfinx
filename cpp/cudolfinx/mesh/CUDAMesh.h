// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cudolfinx/common/CUDA.h>
#include <dolfinx/mesh/Mesh.h>
#include <cudolfinx/mesh/CUDAMeshEntities.h>
#include <cuda.h>
#include <vector>

namespace dolfinx {
namespace mesh {

/// A wrapper for mesh data that is stored in the device memory of a
/// CUDA device.
template <std::floating_point T>
class CUDAMesh
{
public:
  /// Create an empty mesh
  CUDAMesh()
    : _tdim()
    , _num_vertices()
    , _num_coordinates_per_vertex()
    , _dvertex_coordinates(0)
    , _num_cells()
    , _num_vertices_per_cell()
    , _dvertex_indices_per_cell(0)
    , _dcell_permutations(0)
    , _mesh_entities()
  {
  }
  //-----------------------------------------------------------------------------
  /// Create a mesh
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] mesh Data structures for mesh topology and geometry
  CUDAMesh(const CUDA::Context& cuda_context, const dolfinx::mesh::Mesh<T>& mesh)
  {
    CUresult cuda_err;
    const char * cuda_err_description;

    _tdim = mesh.topology()->dim();

    // Allocate device-side storage for vertex coordinates
    auto vertex_coordinates = mesh.geometry().x();
    _num_vertices = vertex_coordinates.size() / 3;
    // TODO figure out how to handle this properly
    // FEniCSx has a dimension of 3 during assembly, but returns a 
    // different value for the dim of mesh.geometry
    _num_coordinates_per_vertex = 3;
    //_num_coordinates_per_vertex = mesh.geometry().dim();
    if (_num_vertices > 0 && _num_coordinates_per_vertex > 0) {
      if (_num_coordinates_per_vertex > 3) {
        throw std::runtime_error(
          "Expected at most 3 coordinates per vertex "
          "instead of " + std::to_string(_num_coordinates_per_vertex) + " "
          "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }

      size_t dvertex_coordinates_size =
        _num_vertices * 3 * sizeof(double);
      cuda_err = cuMemAlloc(
        &_dvertex_coordinates,
        dvertex_coordinates_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemAlloc() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }

      // Copy vertex coordinates to device
      cuda_err = cuMemcpyHtoD(
        _dvertex_coordinates,
        vertex_coordinates.data(),
        dvertex_coordinates_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuMemFree(_dvertex_coordinates);
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
    }

    // Obtain mesh geometry
    auto x_dofmap =
      mesh.geometry().dofmap();

    // Allocate device-side storage for cell vertex indices
    _num_cells = x_dofmap.extent(0);
    _num_vertices_per_cell = x_dofmap.extent(1);
    if (_num_cells > 0 && _num_vertices_per_cell > 0) {
      size_t dvertex_indices_per_cell_size =
        _num_cells * _num_vertices_per_cell * sizeof(int32_t);
      cuda_err = cuMemAlloc(
        &_dvertex_indices_per_cell,
        dvertex_indices_per_cell_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuMemFree(_dvertex_coordinates);
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemAlloc() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }

      // Copy cell vertex indices to device
      cuda_err = cuMemcpyHtoD(
        _dvertex_indices_per_cell,
        x_dofmap.data_handle(),
        dvertex_indices_per_cell_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuMemFree(_dvertex_indices_per_cell);
        cuMemFree(_dvertex_coordinates);
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
    }

    // Obtain cell permutations
    mesh.topology_mutable()->create_entity_permutations();
    auto cell_permutations = mesh.topology()->get_cell_permutation_info();

    // Allocate device-side storage for cell permutations
    if (_num_cells > 0) {
      size_t dcell_permutations_size =
        _num_cells * sizeof(uint32_t);
      cuda_err = cuMemAlloc(
        &_dcell_permutations,
        dcell_permutations_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuMemFree(_dvertex_indices_per_cell);
        cuMemFree(_dvertex_coordinates);
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemAlloc() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }

      // Copy cell permutations to device
      cuda_err = cuMemcpyHtoD(
        _dcell_permutations,
        cell_permutations.data(),
        dcell_permutations_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuMemFree(_dcell_permutations);
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
    }

    for (int dim = 0; dim < _tdim; dim++) {
      _mesh_entities.emplace_back(
        cuda_context, mesh, dim);
    }
  }
  //-----------------------------------------------------------------------------
  /// Destructor
  ~CUDAMesh()
  {
    if (_dcell_permutations)
      cuMemFree(_dcell_permutations);
    if (_dvertex_indices_per_cell)
      cuMemFree(_dvertex_indices_per_cell);
    if (_dvertex_coordinates)
      cuMemFree(_dvertex_coordinates);
  }
  //-----------------------------------------------------------------------------
  /// Copy constructor
  /// @param[in] mesh The object to be copied
  CUDAMesh(const CUDAMesh& mesh) = delete;

  /// Move constructor
  /// @param[in] mesh The object to be moved
  CUDAMesh(CUDAMesh&& mesh)
    : _tdim(mesh._tdim)
    , _num_vertices(mesh._num_vertices)
    , _num_coordinates_per_vertex(mesh._num_coordinates_per_vertex)
    , _dvertex_coordinates(mesh._dvertex_coordinates)
    , _num_cells(mesh._num_cells)
    , _num_vertices_per_cell(mesh._num_vertices_per_cell)
    , _dvertex_indices_per_cell(mesh._dvertex_indices_per_cell)
    , _dcell_permutations(mesh._dcell_permutations)
    , _mesh_entities(std::move(mesh._mesh_entities))
  {
    mesh._tdim = 0;
    mesh._num_vertices = 0;
    mesh._num_coordinates_per_vertex = 0;
    mesh._dvertex_coordinates = 0;
    mesh._num_cells = 0;
    mesh._num_vertices_per_cell = 0;
    mesh._dvertex_indices_per_cell = 0;
    mesh._dcell_permutations = 0;
  }
  //-----------------------------------------------------------------------------
  /// Assignment operator
  /// @param[in] mesh Another CUDAMesh object
  CUDAMesh& operator=(const CUDAMesh& mesh) = delete;

  /// Move assignment operator
  /// @param[in] mesh Another CUDAMesh object
  CUDAMesh& operator=(CUDAMesh&& mesh)
  {
    _tdim = mesh._tdim;
    _num_vertices = mesh._num_vertices;
    _num_coordinates_per_vertex = mesh._num_coordinates_per_vertex;
    _dvertex_coordinates = mesh._dvertex_coordinates;
    _num_cells = mesh._num_cells;
    _num_vertices_per_cell = mesh._num_vertices_per_cell;
    _dvertex_indices_per_cell = mesh._dvertex_indices_per_cell;
    _dcell_permutations = mesh._dcell_permutations;
    _mesh_entities = std::move(mesh._mesh_entities);
    mesh._tdim = 0;
    mesh._num_vertices = 0;
    mesh._num_coordinates_per_vertex = 0;
    mesh._dvertex_coordinates = 0;
    mesh._num_cells = 0;
    mesh._num_vertices_per_cell = 0;
    mesh._dvertex_indices_per_cell = 0;
    mesh._dcell_permutations = 0;
    return *this;
  }
  //-----------------------------------------------------------------------------


  /// Get the topological dimension of the mesh
  int32_t tdim() const { return _tdim; }

  /// Get the number of vertices
  int32_t num_vertices() const { return _num_vertices; }

  /// Get the number of coordinates per vertex
  int32_t num_coordinates_per_vertex() const {
    return _num_coordinates_per_vertex; }

  /// Get a handle to the device-side vertex coordinates
  CUdeviceptr vertex_coordinates() const {
    return _dvertex_coordinates; }

  /// Get the number of cells
  int32_t num_cells() const { return _num_cells; }

  /// Get the number of vertices per cell
  int32_t num_vertices_per_cell() const {
    return _num_vertices_per_cell; }

  /// Get a handle to the device-side cell vertex indices
  CUdeviceptr vertex_indices_per_cell() const {
    return _dvertex_indices_per_cell; }

  /// Get a handle to the device-side cell permutations
  CUdeviceptr cell_permutations() const {
    return _dcell_permutations; }

  /// Get the mesh entities of each dimension
  const std::vector<CUDAMeshEntities<T>>& mesh_entities() const {
    return _mesh_entities; }

private:
  /// The topological dimension of the mesh, or the largest dimension
  /// of any of the mesh entities
  int32_t _tdim;

  /// The number of vertices in the mesh
  int32_t _num_vertices;

  /// The number of coordinates for each vertex
  int32_t _num_coordinates_per_vertex;

  /// The coordinates of the mesh vertices
  CUdeviceptr _dvertex_coordinates;

  /// The number of cells in the mesh
  int32_t _num_cells;

  /// The number of vertices in each cell
  int32_t _num_vertices_per_cell;

  /// The vertex indices of each cell
  CUdeviceptr _dvertex_indices_per_cell;

  /// Cell permutations
  CUdeviceptr _dcell_permutations;

  /// The mesh entities of each dimension
  std::vector<CUDAMeshEntities<T>> _mesh_entities;
};

} // namespace mesh
} // namespace dolfinx

