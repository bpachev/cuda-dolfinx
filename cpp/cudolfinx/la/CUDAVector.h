// Copyright (C) 2020 James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cudolfinx/common/CUDA.h>
#include <cuda.h>
#include <petscvec.h>

namespace dolfinx::la
{

/// A wrapper for a dense vector that is stored in the device memory
/// of a CUDA device.

class CUDAVector
{
public:
  /// Create an empty CUDA vector
  CUDAVector();

  /// Create a vector from a PETSc Vec object
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] x PETSc vector to copy to the device
  /// @param[in] page_lock_values Whether or not to use page-locked
  ///                             memory for the host-side array of
  ///                             values.
  CUDAVector(
      const CUDA::Context& cuda_context,
      Vec x,
      bool page_lock_values = true,
      bool include_ghosts = true);


  /*template <dolfinx::scalar T>
  CUDAVector(const CUDA::Context& cuda_context, std::shared_ptr<la::Vector<T>> x)
   : CUDAVector(cuda_context, la::petsc::create_vector_wrap(*x))
  {
  }*/

  /// Destructor
  ~CUDAVector();

  /// Copy constructor
  /// @param[in] vector The object to be copied
  CUDAVector(const CUDAVector& vector) = delete;

  /// Move constructor
  /// @param[in] vector The object to be moved
  CUDAVector(CUDAVector&& vector);

  /// Assignment operator
  /// @param[in] vector Another CUDAVector object
  CUDAVector& operator=(const CUDAVector& vector) = delete;

  /// Move assignment operator
  /// @param[in] vector Another CUDAVector object
  CUDAVector& operator=(CUDAVector&& vector);

  /// Get a handle to the underlying PETSc vector object
  const Vec vector() const { return _x; }

  /// Get the number of vector values
  int32_t num_values() const { return _num_values; }

  /// Get the number of local vector values
  int32_t num_local_values() const { return _num_local_values; }

  /// Get the number of local vector values
  int32_t num_local_ghosted_values() const { return _num_local_ghosted_values; }

  bool ghosted() const;

  /// Get a handle to the device-side non-zero values
  CUdeviceptr values() const;

  /// Return a handle to the device-side non-zero values
  void restore_values() const;

  /// Get a handle to the device-side non-zero values
  CUdeviceptr values_write() const;

  /// Return a handle to the device-side non-zero values
  void restore_values_write() const;

  /// Update the device-side vector values from the underlying PETSc
  /// vector. If the PETSc vector resides in host memory, then the
  /// values are copied from host memory to device memory. This does
  /// nothing if the PETSc vector is already held in device memory.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  void copy_vector_values_to_device(
    const CUDA::Context& cuda_context);

  /// Update the values of the underlying PETSc vector. If the PETSc
  /// vector resides in host memory, then the values are copied from
  /// device memory to host memory. This does nothing if the PETSc
  /// vector is already held in device memory.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  void copy_vector_values_to_host(
    const CUDA::Context& cuda_context);

  /// Update the device-side values of ghost nodes from the underlying
  /// PETSc vector. If the PETSc vector resides in host memory, then
  /// values are copied from host memory to device memory. This does
  /// nothing if the PETSc vector is already held in device memory.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  void copy_ghost_values_to_device(
    const CUDA::Context& cuda_context);

  /// Update the values of ghost nodes of the underlying PETSc vector.
  /// If the PETSc vector resides in host memory, then ghost values
  /// are copied from device memory to host memory. This does nothing
  /// if the PETSc vector is already held in device memory.
  ///
  /// @param[in] cuda_context A context for a CUDA device
  void copy_ghost_values_to_host(
    const CUDA::Context& cuda_context);

  /// Update vector entries that are owned by this process, but are
  /// represented as ghost values on other processes.
  void apply_ghosts(
    const CUDA::Context& cuda_context);

  /// Update vector entries corresponding to ghost values, meaning
  /// that ghost values are gathered from other processes that own
  /// them.
  bool update_ghosts(
    const CUDA::Context& cuda_context);

private:
  /// Handle to the corresponding PETSc vector object
  Vec _x;

  /// Handle to the corresponding local PETSc vector object, if the
  /// vector is distributed.
  Vec _x_local;

  /// Whether or not the host-side array of values uses page-locked or
  /// pinned memory
  bool _values_page_locked;

  bool _include_ghosts;

  /// The number of values in the global vector
  int32_t _num_values;

  /// The number of values owned by the current MPI rank
  int32_t _num_local_values;

  /// The number of values owned by the current MPI rank
  int32_t _num_local_ghosted_values;

  /// The first value owned by the current MPI rank
  int32_t _local_values_start;

  /// The last value owned by the current MPI rank
  int32_t _local_values_end;

  /// Device-side storage for non-zero values
  mutable CUdeviceptr _dvalues;

  /// Whether or not the device-side pointer is owned by PETSc and
  /// needs to be returned when we are done, or if it was allocated
  /// with cuMemAlloc() and needs to be freed with cuMemFree().
  bool _dvalues_petsc_owned;

public:
  bool debug;
};

} // namespace dolfinx::la
