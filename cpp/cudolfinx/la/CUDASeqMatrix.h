// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cudolfinx/common/CUDA.h>
#include <cuda.h>
#include <petscmat.h>

namespace dolfinx::la
{

/// A wrapper for a matrix in the compressed sparse row (CSR) format
/// that is stored in the device memory of a CUDA device.

class CUDASeqMatrix
{
public:
  /// Create an empty CUDA matrix
  CUDASeqMatrix();

  /// Create a matrix from a PETSc Mat object. Note that the Mat must
  /// be of type MATSEQAIJ.
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
  CUDASeqMatrix(
    const CUDA::Context& cuda_context,
    Mat A,
    bool page_lock_values,
    bool use_seqaijcusparsegetarray);

  /// Destructor
  ~CUDASeqMatrix();

  /// Copy constructor
  /// @param[in] matrix The object to be copied
  CUDASeqMatrix(const CUDASeqMatrix& matrix) = delete;

  /// Move constructor
  /// @param[in] matrix The object to be moved
  CUDASeqMatrix(CUDASeqMatrix&& matrix);

  /// Assignment operator
  /// @param[in] matrix Another CUDASeqMatrix object
  CUDASeqMatrix& operator=(const CUDASeqMatrix& matrix) = delete;

  /// Move assignment operator
  /// @param[in] matrix Another CUDASeqMatrix object
  CUDASeqMatrix& operator=(CUDASeqMatrix&& matrix);

  /// Get the underlying PETSc matrix object
  Mat mat() { return _A; }

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

  /// Get a handle to the device-side row pointers
  CUdeviceptr row_ptr() const { return _drow_ptr; }

  /// Get the number of local non-zeros
  int32_t num_local_nonzeros() const { return _num_local_nonzeros; }

  /// Get a handle to the device-side column indices
  CUdeviceptr column_indices() const { return _dcolumn_indices; }

  /// Get a handle to the device-side non-zero values
  CUdeviceptr values() const;

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

  /// Whether or not the host-side array of non-zero values uses
  /// page-locked or pinned memory
  bool _values_page_locked;

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

  /// Device-side storage for row pointers
  CUdeviceptr _drow_ptr;

  /// The number of non-zeros in the global matrix
  int32_t _num_local_nonzeros;

  /// Device-side storage for column indices
  CUdeviceptr _dcolumn_indices;

  /// Device-side storage for non-zero values
  CUdeviceptr _dvalues;

  /// Whether or not the device-side pointer is owned by PETSc and
  /// needs to be returned when we are done, or if it was allocated
  /// with cuMemAlloc() and needs to be freed with cuMemFree().
  ///
  /// For now, PETSc does not provide access to device-side non-zero
  /// values, even for matrices that are stored on a CUDA
  /// device. Consequently, `_dvalues_petsc_owned` is always false,
  /// and there is potentially some unnecessary copying between the
  /// host and device.
  bool _dvalues_petsc_owned;
};

} // namespace dolfinx::la
