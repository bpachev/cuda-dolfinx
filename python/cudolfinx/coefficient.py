from cudolfinx import cpp as _cucpp
from dolfinx.fem.function import Function
from cudolfinx.context import get_cuda_context
import numpy as np

class Coefficient:
    def __init__(self,
                 f: Function):
        self._ctx = get_cuda_context()

        def functiontype(dtype):
            if np.issubdtype(dtype, np.float32):
                return _cucpp.fem.CUDACoefficient_float32
            elif np.issubdtype(dtype, np.float64):
                return _cucpp.fem.CUDACoefficient_float64
            else:
                raise NotImplementedError(f"Type {dtype} not supported.")

        self._cpp_object = functiontype(f.dtype)(f._cpp_object)

    def interpolate(self,
                    d_g):
        return self._cpp_object.interpolate(d_g._cpp_object)

    def values(self):
        return self._cpp_object.values()
