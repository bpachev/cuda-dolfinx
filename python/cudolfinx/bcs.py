# Copyright (C) 2024 Benjamin Pachev
#
# This file is part of cuDOLFINX
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from cudolfinx import cpp as _cucpp
from dolfinx import cpp as _cpp
from dolfinx.fem.bcs import DirichletBC
import typing

class CUDADirichletBC:
  """Represents a collection of boundary conditions
  """

  def __init__(self, ctx, bcs: typing.List[DirichletBC]):
    """Initialize a collection of boundary conditions
    """

    self._bcs = [bc._cpp_object for bc in bcs]
    self._function_spaces = []
    self._cpp_bc_objects = []
    self._ctx = ctx
    # Prepopulate cache of CUDADirichletBC objects
    for bc in bcs:
        self._get_cpp_bcs(bc.function_space)

  def _get_cpp_bcs(self, V: typing.Union[_cpp.fem.FunctionSpace_float32, _cpp.fem.FunctionSpace_float64]):
    """Create cpp CUDADirichletBC object
    """

    # Use this to avoid needing hashes (which might not be supported)
    # Usually there will be a max of two function spaces associated with a set of bcs
    for i, W in enumerate(self._function_spaces):
      if W == V:
        return self._cpp_bc_objects[i]

    if type(V) is _cpp.fem.FunctionSpace_float32:
      _cpp_bc_obj = _cucpp.fem.CUDADirichletBC_float32(self._ctx, V, self._bcs)
    elif type(V) is _cpp.fem.FunctionSpace_float64:
      _cpp_bc_obj = _cucpp.fem.CUDADirichletBC_float64(self._ctx, V, self._bcs)
    else:
      raise TypeError(f"Invalid type for cpp FunctionSpace object '{type(V)}'")

    self._function_spaces.append(V)
    self._cpp_bc_objects.append(_cpp_bc_obj)
    return _cpp_bc_obj

  def update(self, bcs: typing.List[DirichletBC]):
    """Update a subset of the boundary conditions.

    Used for cases with time-varying boundary conditions whose device-side values
    need to be updated.
    """

    _bcs_to_update = [bc._cpp_object for bc in bcs]
    for _cpp_bc, V in zip(self._cpp_bc_objects, self._function_spaces):
      # filter out anything not contained in the right function space
      _cpp_bc.update([_bc for _bc in _bcs_to_update if V.contains(_bc.function_space)])


