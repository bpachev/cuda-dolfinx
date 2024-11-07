// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of cuDOLFINX
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <cudolfinx/la/petsc.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/utils.h>

Mat la::petsc::create_cuda_matrix(MPI_Comm comm, const SparsityPattern& sp)
{
  PetscErrorCode ierr;
  Mat A;

  // Get IndexMaps from sparsity patterm, and block size
  std::array maps = {sp.index_map(0), sp.index_map(1)};
  const std::array bs = {sp.block_size(0), sp.block_size(1)};
  dolfinx::common::IndexMap col_map = sp.column_index_map();

  // Get global and local dimensions
  const std::int64_t M = bs[0] * maps[0]->size_global();
  const std::int64_t N = bs[1] * maps[1]->size_global();
  const std::int32_t m = bs[0] * maps[0]->size_local();
  const std::int32_t n = bs[1] * maps[1]->size_local();

  // Build data to initialise sparsity pattern (modify for block size)
  std::vector<PetscInt> _row_ptr;
  // Need to ensure correct int type. . .
  std::vector<std::int32_t> _column_indices;
  auto [_edges, _offsets]  = sp.graph();

  // The CUDA assembly kernels aren't currently robust to matrices with variable block size
  // So for now always unroll to 1
  _row_ptr.resize(m+1);
  _row_ptr[0] = 0;
  _column_indices.resize(_edges.size()*bs[0]*bs[1]);
  // index indicating where we are in _edges
  std::size_t edge_index = 0;
  std::size_t unrolled_edge_index = 0;
  // Iterate over (blocked) rows
  for (std::size_t row = 0; row < maps[0]->size_local(); row++) {
    // TODO test with differing block sizes to ensure this is still valid
    PetscInt row_nnz = _offsets[row+1] - _offsets[row];
    PetscInt unrolled_row_nnz = row_nnz * bs[1];

    // row ptr
    for (std::size_t unrolled_row = bs[0]*row; unrolled_row < bs[0]*(row+1); unrolled_row++)
      _row_ptr[unrolled_row+1] = _row_ptr[unrolled_row] + unrolled_row_nnz;

    for (std::size_t j = 0; j < row_nnz; j++) {
      for (std::size_t k = 0; k < bs[1]; k++)
        _column_indices[unrolled_edge_index + j*bs[1] + k] = bs[1]*_edges[edge_index+j] + k;
    }
    // Unroll row block 
    for (std::size_t l = 1; l < bs[0]; l++)
      std::copy_n(std::next(_column_indices.begin(), unrolled_edge_index), unrolled_row_nnz, 
		      std::next(_column_indices.begin(), unrolled_edge_index + l*unrolled_row_nnz));

    edge_index += row_nnz;
    unrolled_edge_index += bs[0] * unrolled_row_nnz;
  }

  // convert local column indices to global ones
  std::vector<std::int64_t> global_column_indices(_column_indices.size());
  col_map.local_to_global(_column_indices, global_column_indices);
  // in case PetscInt won't convert
  std::vector<PetscInt> converted_column_indices(_column_indices.size());
  for (std::size_t i = 0; i < global_column_indices.size(); i++) {
    converted_column_indices[i] = global_column_indices[i];
  }
  MatCreateMPIAIJWithArrays(comm, m, n, M, N, _row_ptr.data(), converted_column_indices.data(), nullptr, &A);
  // Change matrix type to CUDA
  ierr = MatSetType(A, MATMPIAIJCUSPARSE);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatSetFromOptions");

  // Set block sizes
  ierr = MatSetBlockSizes(A, 1, 1);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatSetBlockSizes");

  // Create PETSc local-to-global map/index sets
  ISLocalToGlobalMapping local_to_global0;
  // create unrolled global indices
  const std::vector map0 = maps[0]->global_indices();
  std::vector<PetscInt> _map0;
  _map0.resize(map0.size() * bs[0]);
  for (size_t i = 0; i < map0.size(); i++)
    for (size_t j = 0; j < bs[0]; j++)
      _map0[i*bs[0] + j] = map0[i]*bs[0] + j;
  //const std::vector<PetscInt> _map0(map0.begin(), map0.end());
 ierr = ISLocalToGlobalMappingCreate(MPI_COMM_SELF, 1, _map0.size(),
                                      _map0.data(), PETSC_COPY_VALUES,
                                      &local_to_global0);

  if (ierr != 0)
    petsc::error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");

  // Check for common index maps
  if (maps[0] == maps[1] and bs[0] == bs[1])
  {
    ierr = MatSetLocalToGlobalMapping(A, local_to_global0, local_to_global0);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "MatSetLocalToGlobalMapping");
  }
  else
  {
    ISLocalToGlobalMapping local_to_global1;
    const std::vector map1 = maps[1]->global_indices();
    std::vector<PetscInt> _map1;
    _map1.resize(map1.size() * bs[1]);
    for (size_t i = 0; i < map1.size(); i++)
      for (size_t j = 0; j < bs[1]; j++)
        _map1[i*bs[1] + j] = map1[i]*bs[1] + j;
    //const std::vector<PetscInt> _map1(map1.begin(), map1.end());
    ierr = ISLocalToGlobalMappingCreate(MPI_COMM_SELF, 1, _map1.size(),
                                        _map1.data(), PETSC_COPY_VALUES,
                                        &local_to_global1);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");
    ierr = MatSetLocalToGlobalMapping(A, local_to_global0, local_to_global1);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "MatSetLocalToGlobalMapping");
    ierr = ISLocalToGlobalMappingDestroy(&local_to_global1);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");
  }

  // Clean up local-to-global 0
  ierr = ISLocalToGlobalMappingDestroy(&local_to_global0);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");

  // Set some options on Mat object
  ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatSetOption");
  ierr = MatSetOption(A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatSetOption");

  return A;
}

