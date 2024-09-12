// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CUDAMatrix.h"
#include <cudolfinx/common/CUDA.h>
#include <cudolfinx/la/CUDASeqMatrix.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/la/petsc.h>
#include <iostream>
#include <vector>
#include <cuda.h>
#include <petscmat.h>

using namespace dolfinx;
using namespace dolfinx::la;

//-----------------------------------------------------------------------------
CUDAMatrix::CUDAMatrix()
  : _A(nullptr)
  , _diag()
  , _offdiag()
  , _dcolmap(0)
  , _num_rows()
  , _num_columns()
  , _local_row_start()
  , _local_row_end()
  , _num_local_rows()
  , _num_local_columns()
  , _num_local_offdiag_columns()
{
}
//-----------------------------------------------------------------------------
CUDAMatrix::CUDAMatrix(
  const CUDA::Context& cuda_context,
  Mat A,
  bool page_lock_values,
  bool use_seqaijcusparsegetarray)
  : _A(A)
  , _diag()
  , _offdiag()
  , _dcolmap(0)
  , _num_rows()
  , _num_columns()
  , _local_row_start()
  , _local_row_end()
  , _num_local_rows()
  , _num_local_columns()
  , _num_local_offdiag_columns()
{
  PetscErrorCode ierr;
  CUresult cuda_err;
  const char * cuda_err_description;

  // Check the type of matrix
  MatType matrix_type;
  ierr = MatGetType(A, &matrix_type);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "MatGetType");

  // Get the number of matrix rows and columns
  ierr = MatGetSize(A, &_num_rows, &_num_columns);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "MatGetSize");

  // Get the number of rows and columns owned by the current MPI process
  ierr = MatGetLocalSize(A, &_num_local_rows, &_num_local_columns);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "MatGetLocalSize");

  // TODO: We might need to do some additional work to handle non-zero
  // local_row_start.
  ierr = MatGetOwnershipRange(A, &_local_row_start, &_local_row_end);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "MatGetOwnershipRange");

  if (strcmp(matrix_type, MATSEQAIJ) == 0 ||
      strcmp(matrix_type, MATSEQAIJCUSPARSE) == 0)
  {
    // A non-distributed matrix only has a diagonal part
    _diag = std::make_unique<CUDASeqMatrix>(
      cuda_context, A, page_lock_values, use_seqaijcusparsegetarray);
  } else if (strcmp(matrix_type, MATMPIAIJ) == 0 ||
             strcmp(matrix_type, MATMPIAIJCUSPARSE) == 0)
  {
    // For a distributed matrix, we obtain local diagonal and
    // off-diagonal blocks using MatMPIAIJGetSeqAIJ().
    Mat Ad, Ao;
    const int * colmap;
    ierr = MatMPIAIJGetSeqAIJ(A, &Ad, &Ao, &colmap);
    if (ierr != 0)
      la::petsc::error(ierr, __FILE__, "MatMPIAIJGetSeqAIJ");
    _diag = std::make_unique<CUDASeqMatrix>(
      cuda_context, Ad, page_lock_values, use_seqaijcusparsegetarray);
    _offdiag = std::make_unique<CUDASeqMatrix>(
      cuda_context, Ao, page_lock_values, use_seqaijcusparsegetarray);

    // Get the number of columns in the off-diagonal part of the local
    // matrix.
    ierr = MatGetLocalSize(Ao, NULL, &_num_local_offdiag_columns);
    if (ierr != 0)
      la::petsc::error(ierr, __FILE__, "MatGetLocalSize");

    // Convert the column map from global numbering to the
    // process-local numbering
    ISLocalToGlobalMapping cmapping;
    ierr = MatGetLocalToGlobalMapping(A, NULL, &cmapping);
    if (ierr != 0)
      la::petsc::error(ierr, __FILE__, "MatGetLocalToGlobalMapping");

    std::vector<std::int32_t> colmap_local(_num_local_offdiag_columns);
    ierr = ISGlobalToLocalMappingApply(
      cmapping, IS_GTOLM_MASK, _num_local_offdiag_columns, colmap,
      NULL, colmap_local.data());
    if (ierr != 0)
      la::petsc::error(ierr, __FILE__, "ISGlobalToLocalMappingApply");

    // Allocate device-side storage for off-diagonal column map
    if (_num_local_offdiag_columns > 0) {
      size_t dcolmap_size = _num_local_offdiag_columns * sizeof(int32_t);
      cuda_err = cuMemAlloc(&_dcolmap, dcolmap_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemAlloc() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }

      // Copy off-diagonal column map to device
      cuda_err = cuMemcpyHtoD(
        _dcolmap, colmap_local.data(), dcolmap_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        cuMemFree(_dcolmap);
        throw std::runtime_error(
          "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
    }

  } else {
    throw std::runtime_error(
      "Unsupported matrix type '" + std::string(matrix_type) + "' "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
}
//-----------------------------------------------------------------------------
CUDAMatrix::~CUDAMatrix()
{
}
//-----------------------------------------------------------------------------
CUDAMatrix::CUDAMatrix(CUDAMatrix&& matrix)
  : _A(matrix._A)
  , _diag(std::move(matrix._diag))
  , _offdiag(std::move(matrix._offdiag))
  , _dcolmap(matrix._dcolmap)
  , _num_rows(matrix._num_rows)
  , _num_columns(matrix._num_columns)
  , _local_row_start(matrix._local_row_start)
  , _local_row_end(matrix._local_row_end)
  , _num_local_rows(matrix._num_local_rows)
  , _num_local_columns(matrix._num_local_columns)
  , _num_local_offdiag_columns(matrix._num_local_offdiag_columns)
{
  matrix._A = nullptr;
  matrix._diag = nullptr;
  matrix._offdiag = nullptr;
  matrix._dcolmap = 0;
  matrix._num_rows = 0;
  matrix._num_columns = 0;
  matrix._local_row_start = 0;
  matrix._local_row_end = 0;
  matrix._num_local_rows = 0;
  matrix._num_local_columns = 0;
  matrix._num_local_offdiag_columns = 0;
}
//-----------------------------------------------------------------------------
CUDAMatrix& CUDAMatrix::operator=(CUDAMatrix&& matrix)
{
  _A = matrix._A;
  _diag = std::move(matrix._diag);
  _offdiag = std::move(matrix._offdiag);
  _dcolmap = matrix._dcolmap;
  _num_rows = matrix._num_rows;
  _num_columns = matrix._num_columns;
  _local_row_start = matrix._local_row_start;
  _local_row_end = matrix._local_row_end;
  _num_local_rows = matrix._num_local_rows;
  _num_local_columns = matrix._num_local_columns;
  _num_local_offdiag_columns = matrix._num_local_offdiag_columns;
  matrix._A = nullptr;
  matrix._diag = nullptr;
  matrix._offdiag = nullptr;
  matrix._dcolmap = 0;
  matrix._num_rows = 0;
  matrix._num_columns = 0;
  matrix._local_row_start = 0;
  matrix._local_row_end = 0;
  matrix._num_local_rows = 0;
  matrix._num_local_columns = 0;
  matrix._num_local_offdiag_columns = 0;
  return *this;
}
//-----------------------------------------------------------------------------
void CUDAMatrix::copy_matrix_values_to_host(
  const CUDA::Context& cuda_context)
{
  if (_diag)
    _diag->copy_matrix_values_to_host(cuda_context);
  if (_offdiag)
    _offdiag->copy_matrix_values_to_host(cuda_context);
}
//-----------------------------------------------------------------------------
void CUDAMatrix::apply(MatAssemblyType type)
{
  PetscErrorCode ierr;
  ierr = MatAssemblyBegin(_A, type);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatAssemblyBegin");
  ierr = MatAssemblyEnd(_A, type);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatAssemblyEnd");
}
//-----------------------------------------------------------------------------
void CUDAMatrix::debug_dump()
{
  if (_diag) {
    std::cout << "Dumping diag matrix." << std::endl;
    _diag->debug_dump();
  }
  if (_offdiag) {
    std::cout << "Dumping offdiag matrix." << std::endl;
    _offdiag->debug_dump();
  }
}

