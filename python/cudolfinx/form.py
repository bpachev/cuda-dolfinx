# Copyright (C) 2024 Benjamin Pachev
#
# This file is part of cuDOLFINX
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import collections
from cudolfinx.context import get_cuda_context
from cudolfinx import cpp as _cucpp, jit
from dolfinx import fem as fe
from dolfinx import cpp as _cpp
import numpy as np
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

class BlockCUDAForm:
    """Data structure containing multiple CUDA forms to be used in block assembly."""

    def __init__(
        self, forms: typing.Union[list[CUDAForm], list[list[CUDAForm]]],
        restrictions: typing.Optional[
            typing.Union[
                list[np.typing.NDArray[np.int32]],
                tuple[list[np.typing.NDArray[np.int32]], list[np.typing.NDArray[np.int32]]]
        ]] = None
    ):
        """Initialize the data structure."""

        self._forms = forms
        self._restrictions = restrictions

        if not len(forms): raise ValueError("Must provide at least one form!")
        if type(forms[0]) is CUDAForm: self._init_vector()
        else: self._init_matrix()

    def _init_vector(self):
        """Initialize vector form."""

        offset = 0
        offsets = [offset]
        for i, form in enumerate(self._forms):
            # note in dolfinx 0.10.0 dofmap is replaced with dofmaps
            # which means this portion will require reworking
            dofmap = form.function_spaces[0].dofmap
            local_size = dofmap.index_map.size_local 
            if self._restrictions is not None:
                restriction_inds = self._restrictions[i]
                local_size = len(restriction_inds[restriction_inds < local_size])
            else:
                restriction_inds = np.arange(local_size, dtype=np.int32)
            form.cuda_form.set_restriction([offset], [restriction_inds])
            offset += local_size * dofmap.index_map_bs
            offsets.append(offset)

        self._offsets = offsets
        comm = self._forms[0].dolfinx_form.mesh.comm
        self._global_size = comm.allreduce(offsets[-1])

    def _init_matrix(self):
        """Initialize matrix form."""

        row_offset = col_offset = 0
        row_offsets = []
        col_offsets = []
        row_restrictions = []
        col_restrictions = []
        # iterate over the first form in each row
        for i, row in enumerate(self._forms):
            dofmap = row[0].function_spaces[0].dofmap
            local_size = dofmap.index_map.size_local
            if self._restrictions is not None:
                restriction_inds = self._restrictions[0][i]
                local_size = len(restriction_inds[restriction_inds<local_size])
            else:
                restriction_inds = np.arange(local_size, dtype=np.int32)
            row_restrictions.append(restriction_inds)
            row_offset += dofmap.index_map_bs * local_size
            row_offsets.append(row_offset)
        # iterate over the first row of forms
        for i, form in enumerate(self._forms[0]):
            dofmap = form.function_spaces[1].dofmap
            local_size = dofmap.index_map.size_local
            if self._restrictions is not None:
                restriction_inds = self._restrictions[1][i]
                local_size = len(restriction_inds[restriction_inds<local_size])
            else:
                restriction_inds = np.arange(local_size, dtype=np.int32)
            col_restrictions.append(restriction_inds) 
            col_offset += dofmap.index_map_bs * local_size
            col_offsets.append(col_offset)

        # restrict forms appropriately
        for i, row in enumerate(self._forms):
            for j, form in enumerate(row):
                form.cuda_form.set_restriction(
                        [row_offsets[i], col_offsets[j]],
                        [row_restrictions[i], col_restrictions[j]]
                        )

    @property
    def forms(self):
        """Return the list of forms."""

        return self._forms

    @property
    def dolfinx_forms(self):
        """Return list of underlying dolfinx forms."""

        return [f.dolfinx_form for f in self._forms]

    @property
    def offsets(self):
        """Return list of offsets."""

        return self._offsets

    @property
    def local_size(self):
        """Return size of local vector."""

        return self._offsets[-1]

    @property
    def global_size(self):
        """Return size of global vector."""

        return self._global_size

def form(
    form: typing.Union[ufl.Form, typing.Iterable[ufl.Form]],
    restriction: typing.Optional[typing.Iterable[np.typing.NDArray[np.int32]]] = None,
    **kwargs):
    """Create a CUDAForm from a ufl form."""

    def _create_form(form):
        """Recursively convert ufl.Forms to CUDAForm."""

        if isinstance(form, ufl.Form):
            dolfinx_form = fe.form(form, **kwargs)
            return CUDAForm(dolfinx_form)
        elif isinstance(form, collections.abc.Iterable):
            return list(map(lambda sub_form: _create_form(sub_form), form))
        else:
            raise TypeError("Expected form to be a ufl.Form or an iterable, got type '{type(form)}'!")

    cuda_form = _create_form(form)
    # TODO: properly handle restriction for a single form
    if isinstance(form, collections.abc.Iterable):
        return BlockCUDAForm(cuda_form, restriction)
    else: return cuda_form

def _create_mesh_on_device(cpp_mesh: typing.Union[_cpp.mesh.Mesh_float32, _cpp.mesh.Mesh_float64], ctx: _cucpp.fem.CUDAContext):
  """Create device-side mesh data
  """

  if type(cpp_mesh) is _cpp.mesh.Mesh_float32:
    return _cucpp.fem.CUDAMesh_float32(ctx, cpp_mesh)
  elif type(cpp_mesh) is _cpp.mesh.Mesh_float64:
    return _cucpp.fem.CUDAMesh_float64(ctx, cpp_mesh)
  else:
    raise ValueError(f"Cannot instantiate CUDAMesh for Mesh of type {type(cpp_mesh)}!")

