# Copyright (C) 2026 Chayanon Wichitrnithed
#
# This file is part of cuDOLFINX
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from __future__ import annotations
from cudolfinx import cpp as _cucpp
from dolfinx.fem.function import Function
from cudolfinx.context import get_cuda_context
import numpy as np

class Coefficient:
    """CUDA wrapper class for dolfinx.fem.Function.
    """
    def __init__(self,
                 f: Function):
        """Initialize with a given dolfinx Function f. Create a copy of
        the global DOF vector on both host and device.
        """
        self._ctx = get_cuda_context()

        def functiontype(dtype):
            if np.issubdtype(dtype, np.float32):
                return _cucpp.fem.CUDACoefficient_float32
            elif np.issubdtype(dtype, np.float64):
                return _cucpp.fem.CUDACoefficient_float64
            else:
                raise NotImplementedError(f"Cannot instantiate Coefficient of type {dtype}.")

        self._cpp_object = functiontype(f.dtype)(f._cpp_object)

    def interpolate(self,
                    coeff0: Coefficient):
        """Interpolate from another Coefficient object, modifying
        the global DOF vector. Both must share the
        same mesh and mapping to reference element.

        Args:
            coeff0: A Coefficient object to interpolate from.
        """
        return self._cpp_object.interpolate(coeff0._cpp_object)

    def values(self) -> np.ndarray:
        """Return a copy of the global DOF vector."""
        return self._cpp_object.values()
