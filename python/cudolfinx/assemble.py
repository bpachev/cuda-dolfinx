# Copyright (C) 2024 Benjamin Pachev, James D. Trotter
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from __future__ import annotations

import collections
import functools
import typing
import tempfile

import dolfinx
from dolfinx import cpp as _cpp
from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.forms import Form
from dolfinx.fem.function import Function, FunctionSpace
from dolfinx import fem as fe
from cudolfinx.context import get_cuda_context
from cudolfinx import cpp as _cucpp
from cudolfinx.bcs import CUDADirichletBC
from cudolfinx.form import CUDAForm
from cudolfinx.la import CUDAMatrix, CUDAVector
from petsc4py import PETSc
import numpy as np


def create_petsc_cuda_vector(L: Form) -> PETSc.Vec:
  """Create PETSc Vector on device
  """

  index_map = L.function_spaces[0].dofmap.index_map
  bs = L.function_spaces[0].dofmap.index_map_bs
  size = (index_map.size_local * bs, index_map.size_global * bs)
  # we need to provide at least the local CPU array
  arr = np.zeros(size[0])
  return PETSc.Vec().createCUDAWithArrays(cpuarray=arr, size=size, bsize=bs, comm=index_map.comm)

class CUDAAssembler:
  """Class for assembly on the GPU
  """

  def __init__(self):
    """Initialize the assembler
    """

    self._ctx = get_cuda_context()
    self._tmpdir = tempfile.TemporaryDirectory()
    self._cpp_object = _cucpp.fem.CUDAAssembler(self._ctx, self._tmpdir.name)

  def assemble_matrix(self,
      a: CUDAForm,
      mat: typing.Optional[_cucpp.fem.CUDAMatrix] = None,
      bcs: typing.Optional[typing.Union[list[DirichletBC], CUDADirichletBC]] = [],
      diagonal: float = 1.0,
      constants: typing.Optional[list] = None,
      coeffs: typing.Optional[list] = None
  ):
    """Assemble bilinear form into a matrix on the GPU.

    Args:
        a: The bilinear form to assemble.
        mat: Matrix in which to store assembled values.
        bcs: Boundary conditions that affect the assembled matrix.
            Degrees-of-freedom constrained by a boundary condition will
            have their rows/columns zeroed and the value ``diagonal``
            set on on the matrix diagonal.
        constants: Constants that appear in the form. If not provided,
            any required coefficients will be computed.
        coeffs: Optional list of form coefficients to repack. If not specified, all will be repacked.
           If an empty list is passed, no repacking will be performed.

    Returns:
        Matrix representation of the bilinear form ``a``.

    Note:
        The returned matrix is not finalised, i.e. ghost values are not
        accumulated.

    """

    if not isinstance(a, CUDAForm):
      raise TypeError("Expected CUDAForm, got '{type(a)}'")

    if mat is None:
      mat = self.create_matrix(a)

    if type(bcs) is list:
      bc_collection = self.pack_bcs(bcs)
    elif type(bcs) is CUDADirichletBC:
      bc_collection = bcs
    else:
      raise TypeError(
        f"Expected either a list of DirichletBC's or a CUDADirichletBC, got '{type(bcs)}'"
      )

    _bc0 = bc_collection._get_cpp_bcs(a.dolfinx_form.function_spaces[0])
    _bc1 = bc_collection._get_cpp_bcs(a.dolfinx_form.function_spaces[1])
    # For now always re-copy to device on assembly
    # This assumes something has changed on the host
    a.to_device() 
    self.pack_coefficients(a, coeffs)

    _cucpp.fem.assemble_matrix_on_device(
       self._ctx, self._cpp_object, a.cuda_form,
       a.cuda_mesh, mat._cpp_object, _bc0, _bc1
    )

    return mat

  def assemble_vector(self,
    b: CUDAForm,
    vec: typing.Optional[CUDAVector] = None,
    constants=None, coeffs=None
  ):
    """Assemble linear form into vector on GPU

    Args:
        b: the linear form to use for assembly
        vec: the vector to assemble into. Created if it doesn't exist
        constants: Form constants
        coeffs: Optional list of form coefficients to repack. If not specified, all will be repacked.
           If an empty list is passed, no repacking will be performed.
    """

    if not isinstance(b, CUDAForm):
      raise TypeError("Expected CUDAForm, got '{type(b)}'")

    if vec is None:
      vec = self.create_vector(b)
    # For now always re-copy to device on assembly
    # This assumes something has changed on the host
    b.to_device() 
    self.pack_coefficients(b, coeffs) 
    _cucpp.fem.assemble_vector_on_device(self._ctx, self._cpp_object, b.cuda_form,
      b.cuda_mesh, vec._cpp_object)
    return vec

  def create_matrix(self, a: CUDAForm) -> CUDAMatrix:
    """Create a CUDAMatrix from a given form
    """
    if not isinstance(a, CUDAForm):
      raise TypeError(f"Expected CUDAForm, got type '{type(a)}'.")
    petsc_mat = _cucpp.fem.petsc.create_cuda_matrix(a.dolfinx_form._cpp_object)
    return CUDAMatrix(self._ctx, petsc_mat)

  def create_vector(self, b: CUDAForm) -> CUDAVector:
    """Create a CUDAVector from a given form
    """
    if not isinstance(b, CUDAForm):
      raise TypeError(f"Expected CUDAForm, got type '{type(b)}'.")
    petsc_vec = create_petsc_cuda_vector(b.dolfinx_form)
    return CUDAVector(self._ctx, petsc_vec)

  def pack_bcs(self, bcs: list[DirichletBC]) -> CUDADirichletBC:
    """Pack boundary conditions into a single object for use in assembly.

    The returned object is of type CUDADirichletBC and can be used in place of a list of
    regular DirichletBCs. This is more efficient when performing multiple operations with the same list of 
    boundary conditions, or when boundary condition values need to change over time.
    """

    return CUDADirichletBC(self._ctx, bcs)

  def pack_coefficients(self, a: CUDAForm, coefficients: typing.Optional[list[Function]]=None):
    """Pack coefficients on device
    """
    if not isinstance(a, CUDAForm):
      raise TypeError(f"Expected CUDAForm, got type '{type(a)}'.")
  
    if coefficients is None:
      _cucpp.fem.pack_coefficients(self._ctx, self._cpp_object, a.cuda_form)
    else:
      # perform a repacking with only the indicated coefficients
      _coefficients = [c._cpp_object for c in coefficients]
      _cucpp.fem.pack_coefficients(self._ctx, self._cpp_object, a.cuda_form, _coefficients)

  def apply_lifting(self,
    b: CUDAVector,
    a: list[CUDAForm],
    bcs: typing.List[typing.Union[list[DirichletBC], CUDADirichletBC]],
    x0: typing.Optional[list[Vector]] = None,
    scale: float = 1.0,
    coeffs: typing.Optional[list[list[Function]]] = None
  ):
    """GPU equivalent of apply_lifting

    Args:
       b: CUDAVector to modify
       a: list of forms to lift
       bcs: list of boundary condition lists
       x0: optional list of shift vectors for lifting
       scale: scale of lifting
       coeffs: coefficients to (re-)pack
    """
 
    if len(a) != len(bcs): raise ValueError("Lengths of forms and bcs must match!")
    if x0 is not None and len(x0) != len(a): raise ValueError("Lengths of forms and x0 must match!") 

    _x0 = [] if x0 is None else [x._cpp_object for x in x0]
    bc_collections = []
    for bc_collection in bcs:
      if type(bc_collection) is list:
        bc_collections.append(self.pack_bcs(bc_collection))
      elif type(bc_collection) is CUDADirichletBC:
        bc_collections.append(bc_collection)
      else:
        raise TypeError(
          f"Expected either a list of DirichletBC's or a CUDADirichletBC, got '{type(bc_collection)}'"
        )

    if coeffs is None:
      coeffs = [None] * len(a)
    elif not len(coeffs):
      coeffs = [[] for form in a]
    
    cuda_forms = []
    cuda_mesh = None
    _bcs = []
    for form, bc_collection, form_coeffs in zip(a, bc_collections, coeffs):
      self.pack_coefficients(form, form_coeffs)
      cuda_forms.append(form.cuda_form)
      if cuda_mesh is None: cuda_mesh = form.cuda_mesh
      _bcs.append(bc_collection._get_cpp_bcs(form.function_spaces[1]))

    _cucpp.fem.apply_lifting_on_device(
      self._ctx, self._cpp_object,
      cuda_forms, cuda_mesh,
      b._cpp_object, _bcs, _x0, scale
    )

  def set_bc(self,
    b: CUDAVector,
    bcs: typing.Union[list[DirichletBC], CUDADirichletBC],
    V: FunctionSpace,
    x0: typing.Optional[Function] = None,
    scale: float = 1.0,
  ):
    """Set boundary conditions on device.

    Args:
     b: vector to modify
     bcs: collection of bcs
     V: FunctionSpace to which bcs apply
     x0: optional shift vector
     scale: scaling factor
    """

    if type(bcs) is list:
      bc_collection = self.pack_bcs(bcs)
    elif type(bcs) is CUDADirichletBC:
      bc_collection = bcs
    else:
      raise TypeError(
        f"Expected either a list of DirichletBC's or a CUDADirichletBC, got '{type(bcs)}'"
      )

    if hasattr(V, '_cpp_object'): _cppV = V._cpp_object
    else: _cppV = V
    _bcs = bc_collection._get_cpp_bcs(_cppV)

    if x0 is None:
      _cucpp.fem.set_bc_on_device(
        self._ctx, self._cpp_object, 
        b._cpp_object, _bcs, scale
      )
    else:
      _cucpp.fem.set_bc_on_device(
        self._ctx, self._cpp_object,
        b._cpp_object, _bcs, x0._cpp_object, scale
      )
