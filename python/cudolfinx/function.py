# Copyright (C) 2026 Chayanon Wichitrnithed, Benjamin Pachev
#
# This file is part of cuDOLFINX
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Wrapper for Function."""

from __future__ import annotations

import numpy as np

import ufl
from cudolfinx import cpp as _cucpp
from cudolfinx.context import get_cuda_context
from dolfinx.fem.function import Function


class CUDAFunction(ufl.Coefficient):
    """CUDA wrapper class for dolfinx.fem.Function."""
    def __init__(self,
                 f: Function):
        """Initialize with a given dolfinx Function f.

        Creates a copy of the global DOF vector on both host and device.
        """
        self._ctx = get_cuda_context()

        def functiontype(dtype):
            if np.issubdtype(dtype, np.float32):
                return _cucpp.fem.CUDACoefficient_float32
            elif np.issubdtype(dtype, np.float64):
                return _cucpp.fem.CUDACoefficient_float64
            else:
                raise NotImplementedError(f"Cannot instantiate CUDAFunction of type {dtype}.")

        # have _cpp_object point to the dolfinx.Function's _cpp_object
        # so that we can use CUDAFunction in a dolfinx form
        self._cpp_object = f._cpp_object
        # store actual C++ CUDAFunction as _cuda_function
        # following the pattern set by CUDAForm
        self._cuda_function = functiontype(f.dtype)(f._cpp_object)
        # Initialize UFL properties
        super().__init__(f.function_space.ufl_function_space())

    def interpolate(self,
                    coeff0: CUDAFunction):
        """Interpolate from another CUDAFunction object.

        Both must share the same mesh and mapping to reference element.

        Args:
            coeff0: A CUDAFunction object to interpolate from.
        """
        return self._cpp_object.interpolate(coeff0._cpp_object)

    def values(self) -> np.ndarray:
        """Return a copy of the global DOF vector."""
        return self._cpp_object.values()
