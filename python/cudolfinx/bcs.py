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

    self.bcs = bcs
    self._function_spaces = []
    self._bc_lists = []
    self._device_bcs = []
    self._ctx = ctx
    
    for bc in bcs:
        V = bc.function_space
        try:
            i = self._function_spaces.index(V)
        except ValueError:
            self._function_spaces.append(V)
            self._bc_lists.append([])
            i = -1
        self._bc_lists[i].append(bc._cpp_object)

    for V, cpp_bcs in zip(self._function_spaces, self._bc_lists):
        _cpp_bc_obj = self._make_device_bc(V, cpp_bcs)
        self._device_bcs.append(_cpp_bc_obj)

  def _make_device_bc(self,
          V: typing.Union[_cpp.fem.FunctionSpace_float32, _cpp.fem.FunctionSpace_float64],
          cpp_bcs: typing.List[typing.Union[_cpp.fem.DirichletBC_float32, _cpp.fem.DirichletBC_float64]]
          ):
      """Create device bc object wrapping a list of bcs for the same function space"""

      if type(V) is _cpp.fem.FunctionSpace_float32:
        return _cucpp.fem.CUDADirichletBC_float32(self._ctx, V, cpp_bcs)
      elif type(V) is _cpp.fem.FunctionSpace_float64:
        return _cucpp.fem.CUDADirichletBC_float64(self._ctx, V, cpp_bcs)
      else:
        raise TypeError(f"Invalid type for cpp FunctionSpace object '{type(V)}'")

  def _get_cpp_bcs(self, V: typing.Union[_cpp.fem.FunctionSpace_float32, _cpp.fem.FunctionSpace_float64]):
    """Get cpp CUDADirichletBC object
    """

    # Use this to avoid needing hashes (which might not be supported)
    # Usually there will be a max of two function spaces associated with a set of bcs
    try:
        i = self._function_spaces.index(V)
        return self._device_bcs[i]
    except ValueError:
        # return empty collection
        return self._make_device_bc(V, [])

  def update(self, bcs: typing.Optional[typing.List[DirichletBC]] = None):
    """Update a subset of the boundary conditions.

    Used for cases with time-varying boundary conditions whose device-side values
    need to be updated. By default all boundary conditions are updated
    """

    if bcs is None:
      bcs = self.bcs
    _bcs_to_update = [bc._cpp_object for bc in bcs]
    
    for _cpp_bc, V in zip(self._device_bcs, self._function_spaces):
      # filter out anything not contained in the right function space
      _cpp_bc.update([_bc for _bc in _bcs_to_update if V.contains(_bc.function_space)])

