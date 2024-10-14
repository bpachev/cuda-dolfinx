# Copyright (C) 2024 Benjamin Pachev
#
# This file is part of cuDOLFINX
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from cudolfinx.context import get_cuda_context
from cudolfinx import cpp as _cucpp, jit
from dolfinx import fem as fe
from dolfinx import cpp as _cpp
import typing
import ufl

class CUDAForm:
    """CUDA wrapper class for a dolfinx.fem.Form
    """
    
    def __init__(self, form: fe.Form):
        """Initialize the wrapper
        """

        self._ctx = get_cuda_context()
        self._cuda_mesh = _create_mesh_on_device(form.mesh, self._ctx)

        self._dolfinx_form = form
        self._wrapped_tabulate_tensors = jit.get_wrapped_tabulate_tensors(form)
        ufcx_form_addr = form.module.ffi.cast("uintptr_t", form.module.ffi.addressof(form.ufcx_form))

        cpp_form = form._cpp_object
        if type(cpp_form) is _cpp.fem.Form_float32:
            form_cls = _cucpp.fem.CUDAForm_float32
        elif type(cpp_form) is _cpp.fem.Form_float64:
            form_cls = _cucpp.fem.CUDAForm_float64
        else:
            raise ValueError(f"Cannot instantiate CUDAForm for Form of type {type(cpp_form)}!")

        _tabulate_tensor_names = []
        _tabulate_tensor_sources = []
        for name, source in self._wrapped_tabulate_tensors:
            _tabulate_tensor_names.append(name)
            _tabulate_tensor_sources.append(source)
        self._cuda_form = form_cls(
                self._ctx,
                cpp_form,
                ufcx_form_addr,
                _tabulate_tensor_names,
                _tabulate_tensor_sources
        )

        # TODO expose these parameters to the user
        self._cuda_form.compile(self._ctx, max_threads_per_block=1024, min_blocks_per_multiprocessor=1)

    def to_device(self):
        """Copy host-side coefficients and constants to the device
        """

        self._cuda_form.to_device(self._ctx)

    @property
    def cuda_form(self):
        """Return the underlying cpp CUDAForm
        """

        return self._cuda_form

    @property
    def cuda_mesh(self):
        """Return the underlying cpp CUDAMesh"""

        return self._cuda_mesh

    @property
    def dolfinx_form(self):
        """Return the underlying Dolfinx form
        """

        return self._dolfinx_form

    @property
    def function_spaces(self):
        """Return a list of FunctionSpaces corresponding to the form
        """

        return self._dolfinx_form.function_spaces

def form(form: ufl.Form, **kwargs):
    """Create a CUDAForm from a ufl form
    """

    if not isinstance(form, ufl.Form):
        raise TypeError("Expected form to be a ufl.Form, got type '{type(form)}'!")

    dolfinx_form = fe.form(form, **kwargs)
    return CUDAForm(dolfinx_form)

def _create_mesh_on_device(cpp_mesh: typing.Union[_cpp.mesh.Mesh_float32, _cpp.mesh.Mesh_float64], ctx: _cucpp.fem.CUDAContext):
  """Create device-side mesh data
  """

  if type(cpp_mesh) is _cpp.mesh.Mesh_float32:
    return _cucpp.fem.CUDAMesh_float32(ctx, cpp_mesh)
  elif type(cpp_mesh) is _cpp.mesh.Mesh_float64:
    return _cucpp.fem.CUDAMesh_float64(ctx, cpp_mesh)
  else:
    raise ValueError(f"Cannot instantiate CUDAMesh for Mesh of type {type(cpp_mesh)}!")

