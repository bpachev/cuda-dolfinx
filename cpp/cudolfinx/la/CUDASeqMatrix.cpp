// Copyright (C) 2021 James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CUDASeqMatrix.h"
#include <cudolfinx/common/CUDA.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/la/petsc.h>
#include <iostream>
#include <cuda.h>
#include <petscmat.h>

#if PETSC_VERSION_MAJOR >= 3 && (PETSC_VERSION_MINOR > 16 || (PETSC_VERSION_MINOR == 16 && PETSC_VERSION_SUBMINOR >= 2))
#define PETSC_HAS_MATSEQAIJCUSPARSEGETARRAY 1
#endif

using namespace dolfinx;
using namespace dolfinx::la;

//-----------------------------------------------------------------------------
CUDASeqMatrix::CUDASeqMatrix()
  : _A(nullptr)
  , _values_page_locked(false)
  , _num_rows()
  , _num_columns()
  , _local_row_start()
  , _local_row_end()
  , _num_local_rows()
  , _num_local_columns()
  , _drow_ptr(0)
  , _num_local_nonzeros()
  , _dcolumn_indices(0)
  , _dvalues(0)
  , _dvalues_petsc_owned(false)
{
}
//-----------------------------------------------------------------------------
CUDASeqMatrix::CUDASeqMatrix(
  const CUDA::Context& cuda_context,
  Mat A,
  bool page_lock_values,
  bool use_seqaijcusparsegetarray)
  : _A(A)
  , _values_page_locked(false)
  , _num_rows()
  , _num_columns()
  , _local_row_start()
  , _local_row_end()
  , _num_local_rows()
  , _num_local_columns()
  , _drow_ptr(0)
  , _num_local_nonzeros()
  , _dcolumn_indices(0)
  , _dvalues(0)
  , _dvalues_petsc_owned(false)
{
  PetscErrorCode ierr;
  CUresult cuda_err;
  const char * cuda_err_description;

  // Check the type of matrix
  MatType matrix_type;
  ierr = MatGetType(A, &matrix_type);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "MatGetType");

  if (strcmp(matrix_type, MATSEQAIJ) != 0 &&
      strcmp(matrix_type, MATSEQAIJCUSPARSE) != 0)
  {
    throw std::runtime_error(
      "Unsupported matrix type '" + std::string(matrix_type) + "' "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }

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

  // Obtain the row pointers and column indices of the matrix
  std::int32_t n;
  const std::int32_t* row_ptr = nullptr;
  const std::int32_t* column_indices = nullptr;
  PetscInt shift = 0;
  PetscBool symmetric = PETSC_FALSE;
  PetscBool inodecompressed = PETSC_FALSE;
  PetscBool status = PETSC_FALSE;
  ierr = MatGetRowIJ(
    _A, shift, symmetric, inodecompressed,
    &n, &row_ptr, &column_indices, &status);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "MatGetRowIJ");
  if (status == PETSC_FALSE) {
    throw std::runtime_error(
      "MatGetRowIJ failed with status PETSC_FALSE "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
  if (n != _num_local_rows) {
    throw std::runtime_error(
      "Expected " + std::to_string(_num_local_rows) + " rows "
      "instead of " + std::to_string(n) + " rows from MatGetRowIJ "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
  _num_local_nonzeros =
    _num_local_rows >= 0 ? row_ptr[_num_local_rows] : 0;

  // Allocate device-side storage for row pointers
  if (_num_local_rows > 0) {
    size_t drow_ptr_size = (_num_local_rows+1) * sizeof(int32_t);
    cuda_err = cuMemAlloc(&_drow_ptr, drow_ptr_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Copy row pointers to device
    cuda_err = cuMemcpyHtoD(_drow_ptr, row_ptr, drow_ptr_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      cuMemFree(_drow_ptr);
      ierr = MatRestoreRowIJ(
        _A, shift, symmetric, inodecompressed,
        &n, &row_ptr, &column_indices, &status);
      if (ierr != 0) {
        std::cerr << "MatRestoreRowIJ failed with " << ierr
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;
      }
      if (status == PETSC_FALSE) {
        std::cerr << "MatRestoreRowIJ failed with status PETSC_FALSE"
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;
      }
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }

  // Allocate device-side storage for column indices
  if (_num_local_nonzeros > 0) {
    size_t dcolumn_indices_size = _num_local_nonzeros * sizeof(int32_t);
    cuda_err = cuMemAlloc(&_dcolumn_indices, dcolumn_indices_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      cuMemFree(_drow_ptr);
      ierr = MatRestoreRowIJ(
        _A, shift, symmetric, inodecompressed,
        &n, &row_ptr, &column_indices, &status);
      if (ierr != 0) {
        std::cerr << "MatRestoreRowIJ failed with " << ierr
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;
      }
      if (status == PETSC_FALSE) {
        std::cerr << "MatRestoreRowIJ failed with status PETSC_FALSE"
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;
      }
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Copy column indices to device
    cuda_err = cuMemcpyHtoD(
      _dcolumn_indices, column_indices, dcolumn_indices_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      cuMemFree(_dcolumn_indices);
      cuMemFree(_drow_ptr);
      ierr = MatRestoreRowIJ(
        _A, shift, symmetric, inodecompressed,
        &n, &row_ptr, &column_indices, &status);
      if (ierr != 0) {
        std::cerr << "MatRestoreRowIJ failed with " << ierr
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;
      }
      if (status == PETSC_FALSE) {
        std::cerr << "MatRestoreRowIJ failed with status PETSC_FALSE"
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;
      }
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }

  // Restore the row pointers and column indices. These arrays may no
  // longer be accessed.
  ierr = MatRestoreRowIJ(
    _A, shift, symmetric, inodecompressed,
    &n, &row_ptr, &column_indices, &status);
  if (ierr != 0) {
    cuMemFree(_dcolumn_indices);
    cuMemFree(_drow_ptr);
    la::petsc::error(ierr, __FILE__, "MatRestoreRowIJ");
  }
  if (status == PETSC_FALSE) {
    cuMemFree(_dcolumn_indices);
    cuMemFree(_drow_ptr);
    throw std::runtime_error(
      "MatRestoreRowIJ failed with status PETSC_FALSE "
      "at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }

  if (use_seqaijcusparsegetarray) {
    if (strcmp(matrix_type, MATSEQAIJCUSPARSE) == 0) {
#if defined(PETSC_HAS_MATSEQAIJCUSPARSEGETARRAY)
      _dvalues_petsc_owned = true;
      return;
#else
      cuMemFree(_dcolumn_indices);
      cuMemFree(_drow_ptr);
      throw std::runtime_error(
        "A version of PETSc with MatSeqAIJCUSPARSEGetArray() is needed"
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
#endif
    } else {
      throw std::runtime_error(
        "Unsupported matrix type '" + std::string(matrix_type) + "'"
        " for `use_seqaijcusparsegetarray()`"
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }

  // Obtain the matrix non-zero values
  PetscScalar* values = nullptr;
  ierr = MatSeqAIJGetArray(_A, &values);
  if (ierr != 0) {
    cuMemFree(_dcolumn_indices);
    cuMemFree(_drow_ptr);
    la::petsc::error(ierr, __FILE__, "MatSeqAIJGetArray");
  }

  // Allocate device-side storage for non-zero values
  if (_num_local_nonzeros > 0) {
    size_t dvalues_size = _num_local_nonzeros * sizeof(PetscScalar);
    cuda_err = cuMemAlloc(&_dvalues, dvalues_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      ierr = MatSeqAIJRestoreArray(_A, &values);
      if (ierr != 0) {
        std::cerr << "MatSeqAIJRestoreArray failed with " << ierr
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;
      }
      cuMemFree(_dcolumn_indices);
      cuMemFree(_drow_ptr);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    _values_page_locked = page_lock_values;
    if (_values_page_locked) {
      // Register the host memory as page-locked before copying. This
      // improves throughput if we later copy matrix values back from
      // the device.
      cuda_err = cuMemHostRegister(values, dvalues_size, 0);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        cuMemFree(_dvalues);
        ierr = MatSeqAIJRestoreArray(_A, &values);
        if (ierr != 0) {
          std::cerr << "MatSeqAIJRestoreArray failed with " << ierr
                    << " at " << __FILE__ << ":" << __LINE__ << std::endl;
        }
        cuMemFree(_dcolumn_indices);
        cuMemFree(_drow_ptr);
        throw std::runtime_error(
          "cuMemHostRegister() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
    }

    // Copy non-zero values to device
    cuda_err = cuMemcpyHtoD(_dvalues, values, dvalues_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      cuda_err = cuMemHostUnregister(values);
      if (cuda_err != CUDA_SUCCESS) {
        const char * cuda_err_description;
        cuGetErrorString(cuda_err, &cuda_err_description);
        std::cerr
          << "cuMemHostUnregister() failed with " << cuda_err_description
          << " at " << __FILE__ << ":" << __LINE__ << std::endl;
      }
      cuMemFree(_dvalues);
      ierr = MatSeqAIJRestoreArray(_A, &values);
      if (ierr != 0) {
        std::cerr << "MatSeqAIJRestoreArray failed with " << ierr
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;
      }
      cuMemFree(_dcolumn_indices);
      cuMemFree(_drow_ptr);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
  }

  // Return the matrix values
  ierr = MatSeqAIJRestoreArray(_A, &values);
  if (ierr != 0) {
    cuMemFree(_dvalues);
    cuMemFree(_dcolumn_indices);
    cuMemFree(_drow_ptr);
    la::petsc::error(ierr, __FILE__, "MatSeqAIJRestoreArray");
  }
}
//-----------------------------------------------------------------------------
CUDASeqMatrix::~CUDASeqMatrix()
{
  PetscErrorCode ierr;
  CUresult cuda_err;
  const char * cuda_err_description;

  // Un-register memory, if it was page-locked
  if (_values_page_locked) {
    PetscScalar* values;
    ierr = MatSeqAIJGetArray(_A, &values);
    if (ierr != 0) {
      std::cerr << "MatSeqAIJGetArray failed with " << ierr
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;
    } else {
      cuda_err = cuMemHostUnregister(values);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        std::cerr
          << "cuMemHostUnregister() failed with " << cuda_err_description
          << " at " << __FILE__ << ":" << __LINE__ << std::endl;
      }
      ierr = MatSeqAIJRestoreArray(_A, &values);
      if (ierr != 0) {
        std::cerr << "MatSeqAIJRestoreArray failed with " << ierr
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;
      }
    }
  }

  if (_dvalues) {
    if (_dvalues_petsc_owned) {
#if defined(PETSC_HAS_MATSEQAIJCUSPARSEGETARRAY)
      ierr = MatSeqAIJCUSPARSERestoreArray(_A, (PetscScalar **) &_dvalues);
      if (ierr != 0) {
        std::cerr << "MatSeqAIJCUSPARSERestoreArray failed with " << ierr
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;
      }
#else
      std::cerr << "Assertion failed \"!_dvalues_petsc_owned\""
          " at " << __FILE__ << ":" << __LINE__ << std::endl;
#endif
    } else {
      cuMemFree(_dvalues);
    }
  }
  if (_dcolumn_indices)
    cuMemFree(_dcolumn_indices);
  if (_drow_ptr)
    cuMemFree(_drow_ptr);
}
//-----------------------------------------------------------------------------
CUDASeqMatrix::CUDASeqMatrix(CUDASeqMatrix&& matrix)
  : _A(matrix._A)
  , _values_page_locked(matrix._values_page_locked)
  , _num_rows(matrix._num_rows)
  , _num_columns(matrix._num_columns)
  , _local_row_start(matrix._local_row_start)
  , _local_row_end(matrix._local_row_end)
  , _num_local_rows(matrix._num_local_rows)
  , _num_local_columns(matrix._num_local_columns)
  , _drow_ptr(matrix._drow_ptr)
  , _num_local_nonzeros(matrix._num_local_nonzeros)
  , _dcolumn_indices(matrix._dcolumn_indices)
  , _dvalues(matrix._dvalues)
  , _dvalues_petsc_owned(matrix._dvalues_petsc_owned)
{
  matrix._A = nullptr;
  matrix._values_page_locked = false;
  matrix._num_rows = 0;
  matrix._num_columns = 0;
  matrix._local_row_start = 0;
  matrix._local_row_end = 0;
  matrix._num_local_rows = 0;
  matrix._num_local_columns = 0;
  matrix._drow_ptr = 0;
  matrix._num_local_nonzeros = 0;
  matrix._dcolumn_indices = 0;
  matrix._dvalues = 0;
  matrix._dvalues_petsc_owned = false;
}
//-----------------------------------------------------------------------------
CUDASeqMatrix& CUDASeqMatrix::operator=(CUDASeqMatrix&& matrix)
{
  _A = matrix._A;
  _values_page_locked = matrix._values_page_locked;
  _num_rows = matrix._num_rows;
  _num_columns = matrix._num_columns;
  _local_row_start = matrix._local_row_start;
  _local_row_end = matrix._local_row_end;
  _num_local_rows = matrix._num_local_rows;
  _num_local_columns = matrix._num_local_columns;
  _drow_ptr = matrix._drow_ptr;
  _num_local_nonzeros = matrix._num_local_nonzeros;
  _dcolumn_indices = matrix._dcolumn_indices;
  _dvalues = matrix._dvalues;
  _dvalues_petsc_owned = matrix._dvalues_petsc_owned;
  matrix._A = nullptr;
  matrix._values_page_locked = false;
  matrix._num_rows = 0;
  matrix._num_columns = 0;
  matrix._local_row_start = 0;
  matrix._local_row_end = 0;
  matrix._num_local_rows = 0;
  matrix._num_local_columns = 0;
  matrix._drow_ptr = 0;
  matrix._num_local_nonzeros = 0;
  matrix._dcolumn_indices = 0;
  matrix._dvalues = 0;
  matrix._dvalues_petsc_owned = false;
  return *this;
}
//-----------------------------------------------------------------------------
CUdeviceptr CUDASeqMatrix::values() const
{
  PetscErrorCode ierr;

  if (!_dvalues_petsc_owned) {
    return _dvalues;

  } else {
#if defined(PETSC_HAS_MATSEQAIJCUSPARSEGETARRAY)
    if (_dvalues)
      return _dvalues;

    ierr = MatSeqAIJCUSPARSEGetArray(_A, (PetscScalar **) &_dvalues);
    if (ierr != 0)
      la::petsc::error(ierr, __FILE__, "MatSeqAIJCUSPARSEGetArray");
    return _dvalues;
#else
    throw std::logic_error(
      "Assertion failed \"!_dvalues_petsc_owned\""
      " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
#endif
  }
}
//-----------------------------------------------------------------------------
void CUDASeqMatrix::copy_matrix_values_to_host(
  const CUDA::Context& cuda_context)
{
  PetscErrorCode ierr;
  CUresult cuda_err;
  const char * cuda_err_description;

  if (!_dvalues_petsc_owned && _num_local_nonzeros > 0) {
    // Obtain the matrix non-zero values
    PetscScalar* values;
    ierr = MatSeqAIJGetArray(_A, &values);
    if (ierr != 0)
      la::petsc::error(ierr, __FILE__, "MatSeqAIJGetArray");

    // Copy device-side global matrix values to the host
    size_t dvalues_size = _num_local_nonzeros * sizeof(PetscScalar);
    cuda_err = cuMemcpyDtoH(values, _dvalues, dvalues_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      ierr = MatSeqAIJRestoreArray(_A, &values);
      if (ierr != 0) {
        std::cerr << "MatSeqAIJRestoreArray failed with " << ierr
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;
      }
      throw std::runtime_error(
        "cuMemcpyDtoH() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Set at least the first row to force PETSc to update its
    // device-side matrix values. Note that this is a "trick" that is
    // needed at the moment when working with matrices of type
    // `MATAIJCUSPARSE`. Otherwise, the values that are returned to
    // PETSc with `MatSeqAIJRestoreArray()` remain on the CPU and are
    // never brought to the CUDA device. If the matrix is used later,
    // for example, in a Krylov solver, then PETSc's old values will
    // be used, which is not what we want.
    //
    // Ideally, PETSc would provide a way of directly obtaining a
    // pointer to its device-side storage for the matrix non-zero
    // values, in the same way as it does for CUDA vectors (see
    // `VecCUDAGetArray()`). That way, we could avoid copying data
    // between the host and device for matrices that are already
    // stored on a device.
    ierr = MatSetValuesRow(_A, 0, values);
    if (ierr != 0) {
      PetscErrorCode ierr2;
      ierr2 = MatSeqAIJRestoreArray(_A, &values);
      if (ierr2 != 0) {
        std::cerr << "MatSeqAIJRestoreArray failed with " << ierr
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;
      }
      la::petsc::error(ierr, __FILE__, "MatSetValuesRow");
    }

    // Return the matrix values
    ierr = MatSeqAIJRestoreArray(_A, &values);
    if (ierr != 0)
      la::petsc::error(ierr, __FILE__, "MatSeqAIJRestoreArray");

  } else if (_dvalues_petsc_owned) {
#if defined(PETSC_HAS_MATSEQAIJCUSPARSEGETARRAY)
    // Return device-side pointer to PETSc
    if (_dvalues) {
      ierr = MatSeqAIJCUSPARSERestoreArray(_A, (PetscScalar **) &_dvalues);
      if (ierr != 0)
        la::petsc::error(ierr, __FILE__, "MatSeqAIJCUSPARSERestoreArray");
    }
#else
    throw std::logic_error(
      "Assertion failed \"!_dvalues_petsc_owned\""
      " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
#endif
  }
}
//-----------------------------------------------------------------------------
void CUDASeqMatrix::apply(MatAssemblyType type)
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
void CUDASeqMatrix::debug_dump()
{
  CUdeviceptr dvals;
  int ierr = MatSeqAIJCUSPARSEGetArray(_A, (PetscScalar **) &dvals);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "MatSeqAIJCUSPARSEGetArray");
  std::cout << "Have dvals" << std::endl;
  size_t vals_size = sizeof(PetscScalar) * _num_local_nonzeros;
  PetscScalar* vals = (PetscScalar*) malloc(vals_size);
  std::cout << "calling cuMemcpy" << std::endl;
  auto cuda_err = cuMemcpyDtoH(vals, dvals, vals_size);
  const char * cuda_err_description;
  if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemcpyDtoH() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
  }
  std::cout << "finished calling cuMemcpy" << std::endl;
  float sum = 0;
  for (int i = 0; i < _num_local_nonzeros; i++) sum += vals[i];
  std::cout << "Dumping CUDASeqMatrix: nnz " << _num_local_nonzeros << " mean " << (sum/_num_local_nonzeros) << std::endl;
  free(vals);
}
