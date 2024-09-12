// Copyright (C) 2020 James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CUDAVector.h"
#include <cudolfinx/common/CUDA.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/la/petsc.h>
#include <iostream>
#include <cuda.h>
#include <petscvec.h>

using namespace dolfinx;
using namespace dolfinx::la;

//-----------------------------------------------------------------------------
CUDAVector::CUDAVector()
  : _x(nullptr)
  , _x_local(nullptr)
  , _values_page_locked(false)
  , _include_ghosts(true)
  , _num_values()
  , _num_local_values()
  , _num_local_ghosted_values()
  , _local_values_start()
  , _local_values_end()
  , _dvalues(0)
  , _dvalues_petsc_owned(false)
  , debug(false)
{
}
//-----------------------------------------------------------------------------
CUDAVector::CUDAVector(
  const CUDA::Context& cuda_context,
  Vec x,
  bool page_lock_values,
  bool include_ghosts)
  : _x(x)
  , _x_local(nullptr)
  , _values_page_locked(false)
  , _include_ghosts(include_ghosts)
  , _num_values()
  , _num_local_values()
  , _num_local_ghosted_values()
  , _local_values_start()
  , _local_values_end()
  , _dvalues(0)
  , _dvalues_petsc_owned(false)
  , debug(false)
{
  PetscErrorCode ierr;
  CUresult cuda_err;
  const char * cuda_err_description;

  // Get the number of vector values
  ierr = VecGetSize(_x, &_num_values);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "VecGetSize");

  // Get the number of values owned by the current MPI rank
  ierr = VecGetLocalSize(_x, &_num_local_values);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "VecGetLocalSize");

  ierr = VecGhostGetLocalForm(_x, &_x_local);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "VecGhostGetLocalForm");

  if (_x_local) {
    ierr = VecGetLocalSize(_x_local, &_num_local_ghosted_values);
    if (ierr != 0)
      la::petsc::error(ierr, __FILE__, "VecGetLocalSize");
  }

  // TODO: We might need to do some additional work to handle non-zero
  // local_values_start.
  ierr = VecGetOwnershipRange(
    _x, &_local_values_start, &_local_values_end);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "VecGetOwnershipRange");

  // Check the type of vector
  VecType vector_type;
  if (_x_local) {
    ierr = VecGetType(_x_local, &vector_type);
    if (ierr != 0)
      la::petsc::error(ierr, __FILE__, "VecGetType");
  } else {
    ierr = VecGetType(_x, &vector_type);
    if (ierr != 0)
      la::petsc::error(ierr, __FILE__, "VecGetType");
  }

  if (strcmp(vector_type, VECCUDA) == 0 ||
      strcmp(vector_type, VECMPICUDA) == 0 ||
      strcmp(vector_type, VECSEQCUDA) == 0)
  {
    _dvalues_petsc_owned = true;
  }

  // Allocate device-side storage for values
  if (!_dvalues_petsc_owned && (
        ((!_x_local || !_include_ghosts) && _num_local_values > 0) ||
        ((_x_local && _include_ghosts) && _num_local_ghosted_values > 0)))
  {
    size_t dvalues_size = _num_local_values * sizeof(PetscScalar);
    if (_x_local && _include_ghosts)
      dvalues_size = _num_local_ghosted_values * sizeof(PetscScalar);
    cuda_err = cuMemAlloc(&_dvalues, dvalues_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemAlloc() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Obtain the vector values for the current MPI rank
    PetscScalar* values;
    if (_x_local && _include_ghosts) {
      ierr = VecGetArray(_x_local, &values);
      if (ierr != 0)
        la::petsc::error(ierr, __FILE__, "VecGetArray");
    } else {
      ierr = VecGetArray(_x, &values);
      if (ierr != 0)
        la::petsc::error(ierr, __FILE__, "VecGetArray");
    }

    _values_page_locked = page_lock_values;
    if (_values_page_locked) {
      // Register the host memory as page-locked before copying. This
      // improves throughput if we later copy vector values back from
      // the device.
      cuda_err = cuMemHostRegister(values, dvalues_size, 0);
      if (cuda_err != CUDA_SUCCESS) {
        cuMemFree(_dvalues);
        if (_x_local && _include_ghosts) {
          ierr = VecRestoreArray(_x_local, &values);
        } else {
          ierr = VecRestoreArray(_x, &values);
        }
        if (ierr != 0) {
          std::cerr << "VecRestoreArray failed with " << ierr
                    << " at " << __FILE__ << ":" << __LINE__ << std::endl;
        }
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemHostRegister() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
    }

    // Copy values to device
    cuda_err = cuMemcpyHtoD(_dvalues, values, dvalues_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuda_err = cuMemHostUnregister(values);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        std::cerr
          << "cuMemHostUnregister() failed with " << cuda_err_description
          << " at " << __FILE__ << ":" << __LINE__ << std::endl;
      }
      cuMemFree(_dvalues);
      if (_x_local && _include_ghosts) {
        ierr = VecRestoreArray(_x_local, &values);
      } else {
        ierr = VecRestoreArray(_x, &values);
      }
      if (ierr != 0) {
        std::cerr << "VecRestoreArray failed with " << ierr
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;
      }
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    if (_x_local && _include_ghosts) {
      ierr = VecRestoreArray(_x_local, &values);
    } else {
      ierr = VecRestoreArray(_x, &values);
    }
    if (ierr != 0) {
      if (_values_page_locked) {
        cuda_err = cuMemHostUnregister(values);
        if (cuda_err != CUDA_SUCCESS) {
          cuGetErrorString(cuda_err, &cuda_err_description);
          std::cerr
            << "cuMemHostUnregister() failed with " << cuda_err_description
            << " at " << __FILE__ << ":" << __LINE__ << std::endl;
        }
      }
      cuMemFree(_dvalues);
      la::petsc::error(ierr, __FILE__, "VecRestoreArray");
    }
  }

  ierr = VecGhostRestoreLocalForm(_x, &_x_local);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "VecGhostRestoreLocalForm");
}
//-----------------------------------------------------------------------------
CUDAVector::~CUDAVector()
{
  PetscErrorCode ierr;
  CUresult cuda_err;
  const char * cuda_err_description;

  if (_values_page_locked) {
    PetscScalar* values;
    if (_x_local && _include_ghosts) {
      ierr = VecGetArray(_x_local, &values);
    } else {
      ierr = VecGetArray(_x, &values);
    }
    if (ierr != 0) {
      std::cerr
        << "VecGetArray failed with " << ierr
        << " at " << __FILE__ << ":" << __LINE__ << std::endl;
    }
    cuda_err = cuMemHostUnregister(values);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      std::cerr
        << "cuMemHostUnregister() failed with " << cuda_err_description
        << " at " << __FILE__ << ":" << __LINE__ << std::endl;
    }
    if (_x_local && _include_ghosts) {
      ierr = VecRestoreArray(_x_local, &values);
    } else {
      ierr = VecRestoreArray(_x, &values);
    }
    if (ierr != 0) {
      std::cerr << "VecRestoreArray failed with " << ierr
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;
    }
  }

  if (_dvalues && !_dvalues_petsc_owned)
    cuMemFree(_dvalues);
}
//-----------------------------------------------------------------------------
CUDAVector::CUDAVector(CUDAVector&& vector)
  : _x(vector._x)
  , _x_local(vector._x_local)
  , _values_page_locked(vector._values_page_locked)
  , _include_ghosts(vector._include_ghosts)
  , _num_values(vector._num_values)
  , _num_local_values(vector._num_local_values)
  , _num_local_ghosted_values(vector._num_local_ghosted_values)
  , _local_values_start(vector._local_values_start)
  , _local_values_end(vector._local_values_end)
  , _dvalues(vector._dvalues)
  , _dvalues_petsc_owned(vector._dvalues_petsc_owned)
  , debug(vector.debug)
{
  vector._x = nullptr;
  vector._x_local = nullptr;
  vector._values_page_locked = false;
  vector._include_ghosts = true;
  vector._num_values = 0;
  vector._num_local_values = 0;
  vector._num_local_ghosted_values = 0;
  vector._local_values_start = 0;
  vector._local_values_end = 0;
  vector._dvalues = 0;
  vector._dvalues_petsc_owned = false;
  vector.debug = false;
}
//-----------------------------------------------------------------------------
CUDAVector& CUDAVector::operator=(CUDAVector&& vector)
{
  _x = vector._x;
  _x_local = vector._x_local;
  _values_page_locked = vector._values_page_locked;
  _include_ghosts = vector._include_ghosts;
  _num_values = vector._num_values;
  _num_local_values = vector._num_local_values;
  _num_local_ghosted_values = vector._num_local_ghosted_values;
  _local_values_start = vector._local_values_start;
  _local_values_end = vector._local_values_end;
  _dvalues = vector._dvalues;
  _dvalues_petsc_owned = vector._dvalues_petsc_owned;
  debug = vector.debug;
  vector._x = nullptr;
  vector._x_local = nullptr;
  vector._values_page_locked = false;
  vector._include_ghosts = true;
  vector._num_values = 0;
  vector._num_local_values = 0;
  vector._num_local_ghosted_values = 0;
  vector._local_values_start = 0;
  vector._local_values_end = 0;
  vector._dvalues = 0;
  vector._dvalues_petsc_owned = false;
  vector.debug = false;
  return *this;
}
//-----------------------------------------------------------------------------
bool CUDAVector::ghosted() const
{
  PetscErrorCode ierr;
  if (_dvalues_petsc_owned || !_include_ghosts) return false;
  Vec xloc;
  ierr = VecGhostGetLocalForm(_x, &xloc);
  if (ierr != 0) la::petsc::error(ierr, __FILE__, "VecGhostGetLocalForm");
  bool ghosted = xloc != NULL;
  ierr = VecGhostRestoreLocalForm(_x, &xloc);
  if (ierr != 0) la::petsc::error(ierr, __FILE__, "VecGhostRestoreLocalForm");
  return ghosted;
}
//-----------------------------------------------------------------------------
CUdeviceptr CUDAVector::values() const
{
  PetscErrorCode ierr;
  if (!_dvalues_petsc_owned) {
    return _dvalues;
  } else {
    if (_dvalues) {
      if (debug) fprintf(stderr, "%s:%d: CUDAVector::values()\n", __FILE__, __LINE__);
      return _dvalues;
    }
    if (debug) fprintf(stderr, "%s:%d: CUDAVector::values() -> VecCUDAGetArrayRead()\n", __FILE__, __LINE__);
    ierr = VecCUDAGetArrayRead(_x, (const PetscScalar **) &_dvalues);
    if (ierr != 0) la::petsc::error(ierr, __FILE__, "VecCUDAGetArrayRead");
    return _dvalues;
  }
}
//-----------------------------------------------------------------------------
void CUDAVector::restore_values() const
{
  PetscErrorCode ierr;
  if (_dvalues_petsc_owned) {
    if (_dvalues) {
      if (debug) fprintf(stderr, "%s:%d: CUDAVector::restore_values() -> VecCUDARestoreArrayRead()\n", __FILE__, __LINE__);
      ierr = VecCUDARestoreArrayRead(_x, (const PetscScalar **) &_dvalues);
      _dvalues = 0;
      if (ierr != 0) la::petsc::error(ierr, __FILE__, "VecCUDARestoreArrayRead");
    } else {
      if (debug) fprintf(stderr, "%s:%d: CUDAVector::restore_values()\n", __FILE__, __LINE__);
    }
  }
}
//-----------------------------------------------------------------------------
CUdeviceptr CUDAVector::values_write() const
{
  PetscErrorCode ierr;
  if (!_dvalues_petsc_owned) {
    return _dvalues;
  } else {
    if (_dvalues) {
      if (debug) fprintf(stderr, "%s:%d: CUDAVector::values()\n", __FILE__, __LINE__);
      return _dvalues;
    }
    if (debug) fprintf(stderr, "%s:%d: CUDAVector::values() -> VecCUDAGetArrayWrite()\n", __FILE__, __LINE__);
    ierr = VecCUDAGetArrayWrite(_x, (PetscScalar **) &_dvalues);
    if (ierr != 0) la::petsc::error(ierr, __FILE__, "VecCUDAGetArray");
    return _dvalues;
  }
}
//-----------------------------------------------------------------------------
void CUDAVector::restore_values_write() const
{
  PetscErrorCode ierr;
  if (_dvalues_petsc_owned) {
    if (_dvalues) {
      if (debug) fprintf(stderr, "%s:%d: CUDAVector::restore_values() -> VecCUDARestoreArrayWrite()\n", __FILE__, __LINE__);
      ierr = VecCUDARestoreArrayWrite(_x, (PetscScalar **) &_dvalues);
      _dvalues = 0;
      if (ierr != 0) la::petsc::error(ierr, __FILE__, "VecCUDARestoreArrayWrite");
    } else {
      if (debug) fprintf(stderr, "%s:%d: CUDAVector::restore_values()\n", __FILE__, __LINE__);
    }
  }
}
//-----------------------------------------------------------------------------
void CUDAVector::apply_ghosts(
  const CUDA::Context& cuda_context)
{
  PetscErrorCode ierr;
  copy_ghost_values_to_host(cuda_context);
  ierr = VecGhostGetLocalForm(_x, &_x_local);
  if (ierr != 0) la::petsc::error(ierr, __FILE__, "VecGhostGetLocalForm");
  if (_x_local) {
    ierr = VecGhostUpdateBegin(_x, ADD_VALUES, SCATTER_REVERSE);
    if (ierr != 0) la::petsc::error(ierr, __FILE__, "VecGhostUpdateBegin");
    ierr = VecGhostUpdateEnd(_x, ADD_VALUES, SCATTER_REVERSE);
    if (ierr != 0) la::petsc::error(ierr, __FILE__, "VecGhostUpdateBegin");
  }
  ierr = VecGhostRestoreLocalForm(_x, &_x_local);
  if (ierr != 0) la::petsc::error(ierr, __FILE__, "VecGhostRestoreLocalForm");
  copy_ghost_values_to_device(cuda_context);
}
//-----------------------------------------------------------------------------
bool CUDAVector::update_ghosts(
  const CUDA::Context& cuda_context)
{
  PetscErrorCode ierr;
  copy_ghost_values_to_host(cuda_context);
  ierr = VecGhostGetLocalForm(_x, &_x_local);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "VecGhostGetLocalForm");
  bool ghosted = (_x_local != nullptr);
  if (_x_local) {
    ierr = VecGhostUpdateBegin(_x, INSERT_VALUES, SCATTER_FORWARD);
    if (ierr != 0) la::petsc::error(ierr, __FILE__, "VecGhostUpdateBegin");
    ierr = VecGhostUpdateEnd(_x, INSERT_VALUES, SCATTER_FORWARD);
    if (ierr != 0) la::petsc::error(ierr, __FILE__, "VecGhostUpdateBegin");
  }
  ierr = VecGhostRestoreLocalForm(_x, &_x_local);
  if (ierr != 0) la::petsc::error(ierr, __FILE__, "VecGhostRestoreLocalForm");
  copy_ghost_values_to_device(cuda_context);
  return ghosted;
}
//-----------------------------------------------------------------------------
void CUDAVector::copy_ghost_values_to_device(
  const CUDA::Context& cuda_context)
{
  PetscErrorCode ierr;
  CUresult cuda_err;
  const char * cuda_err_description;

  if (_dvalues_petsc_owned) {
    // PetscScalar* values;
    // ierr = VecGetArray(_x, &values);
    // if (ierr != 0) la::petsc::error(ierr, __FILE__, "VecGetArray");
    // ierr = VecRestoreArray(_x, &values);
    // if (ierr != 0) la::petsc::error(ierr, __FILE__, "VecRestoreArray");
    return;
  }

  if (!_include_ghosts || _num_local_ghosted_values <= _num_local_values)
    return;

  ierr = VecGhostGetLocalForm(_x, &_x_local);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "VecGhostGetLocalForm");

  if (_x_local) {
    PetscScalar* values;
    ierr = VecGetArray(_x_local, &values);
    if (ierr != 0)
      la::petsc::error(ierr, __FILE__, "VecGetArray");

    // Copy device-side vector values to the host
    size_t dvalues_size = (_num_local_ghosted_values - _num_local_values) *
      sizeof(PetscScalar);
    cuda_err = cuMemcpyHtoD(
      (CUdeviceptr) ((char *) _dvalues + _num_local_values * sizeof(PetscScalar)),
      &values[_num_local_values],
      dvalues_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    ierr = VecRestoreArray(_x_local, &values);
    if (ierr != 0)
      la::petsc::error(ierr, __FILE__, "VecRestoreArray");
  }

  ierr = VecGhostRestoreLocalForm(_x, &_x_local);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "VecGhostRestoreLocalForm");
}
//-----------------------------------------------------------------------------
void CUDAVector::copy_ghost_values_to_host(
  const CUDA::Context& cuda_context)
{
  PetscErrorCode ierr;
  CUresult cuda_err;
  const char * cuda_err_description;

  if (_dvalues_petsc_owned) {
    // if (debug) fprintf(stderr, "%s:%d: CUDAVector::copy_ghost_values_to_host() -> VecCUDAGetArray()\n", __FILE__, __LINE__);
    // ierr = VecCUDAGetArray(_x, (PetscScalar **) &_dvalues);
    // if (ierr != 0) la::petsc::error(ierr, __FILE__, "VecCUDAGetArray");
    // if (debug) fprintf(stderr, "%s:%d: CUDAVector::copy_ghost_values_to_host() -> VecCUDARestoreArray()\n", __FILE__, __LINE__);
    // ierr = VecCUDARestoreArray(_x, (PetscScalar **) &_dvalues);
    // if (ierr != 0) la::petsc::error(ierr, __FILE__, "VecCUDARestoreArray");
    // _dvalues = 0;
    return;
  }

  if (!_include_ghosts || _num_local_ghosted_values <= _num_local_values)
    return;

  ierr = VecGhostGetLocalForm(_x, &_x_local);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "VecGhostGetLocalForm");

  if (_x_local) {
    PetscScalar* values;
    ierr = VecGetArray(_x_local, &values);
    if (ierr != 0)
      la::petsc::error(ierr, __FILE__, "VecGetArray");

    // Copy device-side vector values to the host
    size_t dvalues_size = (_num_local_ghosted_values - _num_local_values) *
      sizeof(PetscScalar);
    cuda_err = cuMemcpyDtoH(
      &values[_num_local_values],
      (CUdeviceptr) (((char *) _dvalues) + _num_local_values * sizeof(PetscScalar)),
      dvalues_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemcpyDtoH() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    ierr = VecRestoreArray(_x_local, &values);
    if (ierr != 0)
      la::petsc::error(ierr, __FILE__, "VecRestoreArray");
  }

  ierr = VecGhostRestoreLocalForm(_x, &_x_local);
  if (ierr != 0)
    la::petsc::error(ierr, __FILE__, "VecGhostRestoreLocalForm");
}
//-----------------------------------------------------------------------------
void CUDAVector::copy_vector_values_to_device(
  const CUDA::Context& cuda_context)
{
  PetscErrorCode ierr;
  CUresult cuda_err;
  const char * cuda_err_description;

  if (!_dvalues_petsc_owned && _num_local_values > 0) {

    if (_include_ghosts) {
      ierr = VecGhostGetLocalForm(_x, &_x_local);
      if (ierr != 0)
        la::petsc::error(ierr, __FILE__, "VecGhostGetLocalForm");
    }

    PetscScalar* values;
    if (_x_local && _include_ghosts) {
      ierr = VecGetArray(_x_local, &values);
    } else {
      ierr = VecGetArray(_x, &values);
    }
    if (ierr != 0)
      la::petsc::error(ierr, __FILE__, "VecGetArray");

    // Copy device-side vector values to the host
    size_t dvalues_size = _num_local_values * sizeof(PetscScalar);
    if (_x_local && _include_ghosts)
      dvalues_size = _num_local_ghosted_values * sizeof(PetscScalar);
    cuda_err = cuMemcpyHtoD(_dvalues, values, dvalues_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    if (_x_local && _include_ghosts) {
      ierr = VecRestoreArray(_x_local, &values);
    } else {
      ierr = VecRestoreArray(_x, &values);
    }
    if (ierr != 0)
      la::petsc::error(ierr, __FILE__, "VecRestoreArray");

    if (_include_ghosts) {
      ierr = VecGhostRestoreLocalForm(_x, &_x_local);
      if (ierr != 0)
        la::petsc::error(ierr, __FILE__, "VecGhostRestoreLocalForm");
    }

  } else if (_dvalues_petsc_owned) {
     PetscScalar* values;
     ierr = VecGetArrayWrite(_x, &values);
     if (ierr != 0) la::petsc::error(ierr, __FILE__, "VecGetArrayWrite");
     ierr = VecRestoreArrayWrite(_x, &values);
     if (ierr != 0) la::petsc::error(ierr, __FILE__, "VecRestoreArrayWrite");
  }
}
//-----------------------------------------------------------------------------
void CUDAVector::copy_vector_values_to_host(
  const CUDA::Context& cuda_context)
{
  PetscErrorCode ierr;
  CUresult cuda_err;
  const char * cuda_err_description;

  if (!_dvalues_petsc_owned && _num_local_values > 0) {

    if (_include_ghosts) {
      ierr = VecGhostGetLocalForm(_x, &_x_local);
      if (ierr != 0)
        la::petsc::error(ierr, __FILE__, "VecGhostGetLocalForm");
    }

    PetscScalar* values;
    if (_x_local && _include_ghosts) {
      ierr = VecGetArray(_x_local, &values);
    } else {
      ierr = VecGetArray(_x, &values);
    }
    if (ierr != 0)
      la::petsc::error(ierr, __FILE__, "VecGetArray");

    // Copy device-side vector values to the host
    size_t dvalues_size = _num_local_values * sizeof(PetscScalar);
    if (_x_local && _include_ghosts)
      dvalues_size = _num_local_ghosted_values * sizeof(PetscScalar);
    cuda_err = cuMemcpyDtoH(values, _dvalues, dvalues_size);
    if (cuda_err != CUDA_SUCCESS) {
      cuGetErrorString(cuda_err, &cuda_err_description);
      throw std::runtime_error(
        "cuMemcpyDtoH() failed with " + std::string(cuda_err_description) +
        " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    if (_x_local && _include_ghosts) {
      ierr = VecRestoreArray(_x_local, &values);
    } else {
      ierr = VecRestoreArray(_x, &values);
    }
    if (ierr != 0)
      la::petsc::error(ierr, __FILE__, "VecRestoreArray");

    if (_include_ghosts) {
      ierr = VecGhostRestoreLocalForm(_x, &_x_local);
      if (ierr != 0)
        la::petsc::error(ierr, __FILE__, "VecGhostRestoreLocalForm");
    }

  } else if (_dvalues_petsc_owned) {
    // if (debug) fprintf(stderr, "%s:%d: CUDAVector::copy_vector_values_to_device() -> VecCUDAGetArray()\n", __FILE__, __LINE__);
    // ierr = VecCUDAGetArray(_x, (PetscScalar **) &_dvalues);
    // if (ierr != 0) la::petsc::error(ierr, __FILE__, "VecCUDAGetArray");
    // if (debug) fprintf(stderr, "%s:%d: CUDAVector::copy_vector_values_to_device() -> VecCUDARestoreArray()\n", __FILE__, __LINE__);
    // ierr = VecCUDARestoreArray(_x, (PetscScalar **) &_dvalues);
    // if (ierr != 0) la::petsc::error(ierr, __FILE__, "VecCUDARestoreArray");
    // _dvalues = 0;
  }
}
//-----------------------------------------------------------------------------
