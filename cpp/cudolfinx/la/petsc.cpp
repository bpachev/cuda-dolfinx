// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
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

  // Get global and local dimensions
  const std::int64_t M = bs[0] * maps[0]->size_global();
  const std::int64_t N = bs[1] * maps[1]->size_global();
  const std::int32_t m = bs[0] * maps[0]->size_local();
  const std::int32_t n = bs[1] * maps[1]->size_local();

  // Find a common block size across rows/columns
  const int _bs = (bs[0] == bs[1] ? bs[0] : 1);

  // Build data to initialise sparsity pattern (modify for block size)
  std::vector<PetscInt> _row_ptr;
  // Need to ensure correct int type. . .
  std::vector<PetscInt> _column_indices;
  auto [_edges, _offsets]  = sp.graph();
  _column_indices.resize(_edges.size());
  for (std::size_t i = 0; i < _column_indices.size(); ++i) {
    _column_indices[i] = _edges[i];
  }

  if (bs[0] == bs[1])
  {
    _row_ptr.resize(maps[0]->size_local() + 1);
    for (std::size_t i = 0; i < _row_ptr.size(); ++i)
      _row_ptr[i] = _offsets[i];
  }
  else
  {
    // Expand for block size 1
    _row_ptr.resize(maps[0]->size_local() * bs[0] + 1);
    _row_ptr[0] = 0;
    for (std::size_t i = 0; i < _row_ptr.size()-1; ++i)
      _row_ptr[i+1] = _row_ptr[i] + bs[1] * (sp.nnz_diag(i / bs[0]) + sp.nnz_off_diag(i / bs[0]));
  }

  MatCreateMPIAIJWithArrays(comm, m, n, M, N, _row_ptr.data(), _column_indices.data(), nullptr, &A);
  // Change matrix type to CUDA
  ierr = MatSetType(A, MATMPIAIJCUSPARSE);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatSetFromOptions");

  // Set block sizes
  ierr = MatSetBlockSizes(A, bs[0], bs[1]);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatSetBlockSizes");

  // Create PETSc local-to-global map/index sets
  ISLocalToGlobalMapping local_to_global0;
  const std::vector map0 = maps[0]->global_indices();
  const std::vector<PetscInt> _map0(map0.begin(), map0.end());
 ierr = ISLocalToGlobalMappingCreate(MPI_COMM_SELF, bs[0], _map0.size(),
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
    const std::vector<PetscInt> _map1(map1.begin(), map1.end());
    ierr = ISLocalToGlobalMappingCreate(MPI_COMM_SELF, bs[1], _map1.size(),
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

