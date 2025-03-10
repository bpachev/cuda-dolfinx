// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of cuDOLFINX
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cudolfinx/common/CUDA.h>
#include <cuda.h>
#include <petscmat.h>
#include <memory>

namespace dolfinx::la
{

class CUDASeqMatrix;

/// A wrapper for a matrix in the compressed sparse row (CSR) format
/// that is stored in the device memory of a CUDA device.

class CUDAMatrix
{
public:
  /// Create an empty CUDA matrix
  CUDAMatrix();

  /// Create a matrix from a PETSc Mat object
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] A PETSc matrix to copy to the device
  /// @param[in] page_lock_values Whether or not to use page-locked
  ///                             memory for the host-side array of
  ///                             non-zero values.
  /// @param[in] use_seqaijcusparsegetarray Whether or not to use the
  ///              function MatSeqAIJCUSPARSEGetArray(),which is only
  ///              available in a custom-built version of PETSc. If it
  ///              is set, this will avoid unnecessary copying of data
  ///              between host and device for matrices of type
  ///              MATSEQAIJCUSPARSE whenever a CUDA-based assembler
  ///              is used.
  CUDAMatrix(
    const CUDA::Context& cuda_context,
    Mat A,
    bool page_lock_values,
    bool use_seqaijcusparsegetarray);

  /// Destructor
  ~CUDAMatrix();

  /// Copy constructor
  /// @param[in] matrix The object to be copied
  CUDAMatrix(const CUDAMatrix& matrix) = delete;

  /// Move constructor
  /// @param[in] matrix The object to be moved
  CUDAMatrix(CUDAMatrix&& matrix);

  /// Assignment operator
  /// @param[in] matrix Another CUDAMatrix object
  CUDAMatrix& operator=(const CUDAMatrix& matrix) = delete;

  /// Move assignment operator
  /// @param[in] matrix Another CUDAMatrix object
  CUDAMatrix& operator=(CUDAMatrix&& matrix);

  /// Get the underlying PETSc matrix object
  Mat mat() { return _A; }

  /// Get the diagonal block of the local part of the matrix
  const CUDASeqMatrix * diag() const { return _diag.get(); }
  CUDASeqMatrix * diag() { return _diag.get(); }

  /// Get the off-diagonal block of the local part of the matrix
  const CUDASeqMatrix * offdiag() const { return _offdiag.get(); }
  CUDASeqMatrix * offdiag() { return _offdiag.get(); }

  /// Methods to get off diagonal column mapping
  CUdeviceptr colmap() const { return _dcolmap; }
  CUdeviceptr colmap_sorted() const { return _dcolmap_sorted; }
  CUdeviceptr colmap_sorted_indices() const { return _dcolmap_sorted_indices; }

  /// Get the number of matrix rows
  int32_t num_rows() const { return _num_rows; }

  /// Get the number of matrix columns
  int32_t num_columns() const { return _num_columns; }

  /// Get the global index of the first row
  int32_t local_row_start() const { return _local_row_start; }

  /// Get the global index of the last row
  int32_t local_row_end() const { return _local_row_end; }

  /// Get the number of local matrix rows
  int32_t num_local_rows() const { return _num_local_rows; }

  /// Get the number of local matrix columns
  int32_t num_local_columns() const { return _num_local_columns; }

  /// Get the number of local matrix columns in the off-diagonal part
  int32_t num_local_offdiag_columns() const { return _num_local_offdiag_columns; }

  /// Update the values of the underlying PETSc matrix by copying
  /// values from device memory to host memory.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  void copy_matrix_values_to_host(
    const CUDA::Context& cuda_context);

  /// Finalize matrix assembly by calling PETSc's MatAssemblyBegin()
  /// and MatAssemblyEnd().
  ///
  /// @param[in] type MAT_FLUSH_ASSEMBLY or MAT_FINAL_ASSEMBLY
  void apply(MatAssemblyType type);

  void debug_dump();

private:
  /// Handle to the corresponding PETSc matrix object
  Mat _A;

  /// The diagonal block of the local part of the matrix
  std::unique_ptr<CUDASeqMatrix> _diag;

  /// The off-diagonal block of the local part of the matrix.
  /// This is only used if the matrix is distributed.
  std::unique_ptr<CUDASeqMatrix> _offdiag;

  /// Device-side mapping from columns of the local, off-diagonal
  /// block of the matrix to columns of the global matrix.
  CUdeviceptr _dcolmap;
  CUdeviceptr _dcolmap_sorted;
  CUdeviceptr _dcolmap_sorted_indices;

  /// The number of rows in the global matrix
  int32_t _num_rows;

  /// The number of columns in the global matrix
  int32_t _num_columns;

  /// The first row owned by the current MPI process
  int32_t _local_row_start;

  /// The last row owned by the current MPI process
  int32_t _local_row_end;

  /// The number of rows owned by the current MPI process
  int32_t _num_local_rows;

  /// The number of columns owned by the current MPI process
  int32_t _num_local_columns;

  /// The number of columns in the off-diagonal part of the local
  /// matrix owned by the current MPI process
  int32_t _num_local_offdiag_columns;
};

} // namespace dolfinx::la
