# Copyright (C) 2024 Benjamin Pachev
#
# This file is part of cuDOLFINX
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Wrapper classes for CUDA matrices and vectors."""

from cudolfinx import cpp as _cucpp

__all__ = [
  "CUDAMatrix",
  "CUDAVector",
]

class CUDAVector:
  """Vector on device
  """

  def __init__(self, ctx, vec):
    """Initialize the vector
    """

    self._petsc_vec = vec
    self._ctx = ctx
    self._cpp_object = _cucpp.fem.CUDAVector(ctx, self._petsc_vec)

  @property
  def vector(self):
    """Return underlying PETSc vector
    """

    return self._petsc_vec

  def to_host(self):
    """Copy device-side values to host
    """

    self._cpp_object.to_host(self._ctx)

  def __del__(self):
    """Delete the vector and free up GPU resources
    """

    # Ensure that the cpp CUDAVector is taken care of BEFORE the petsc vector. . . .
    del self._cpp_object

class CUDAMatrix:
  """Matrix on device
  """

  def __init__(self, ctx, petsc_mat):
    """Initialize the matrix
    """

    self._petsc_mat = petsc_mat
    self._ctx = ctx
    self._cpp_object = _cucpp.fem.CUDAMatrix(ctx, petsc_mat)

  @property
  def mat(self):
    """Return underlying CUDA matrix
    """

    return self._petsc_mat

  def assemble(self):
    """Call assemble on the underlying PETSc matrix.

    If the PETSc matrix is not a CUDA matrix, then matrix
    values will be explicitly copied to the host.
    """

    self._cpp_object.to_host(self._ctx)

  def __del__(self):
    """Delete the matrix and free up GPU resources
    """

    # make sure we delete the CUDAMatrix before the petsc matrix
    del self._cpp_object


