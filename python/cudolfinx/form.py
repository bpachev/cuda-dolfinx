
from cudolfinx import cpp as _cucpp
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

        self._ctx = _cucpp.fem.CUDAContext()
        self._cuda_mesh = _create_mesh_on_device(form.mesh, self._ctx)

        self._dolfinx_form = form
        ufcx_form_addr = form.module.ffi.cast("uintptr_t", form.module.ffi.addressof(form.ufcx_form))

        cpp_form = form._cpp_object
        if type(cpp_form) is _cpp.fem.Form_float32:
            self._cuda_form = _cucpp.fem.CUDAForm_float32(self._ctx, cpp_form, ufcx_form_addr)
        elif type(cpp_form) is _cpp.fem.Form_float64:
            self._cuda_form = _cucpp.fem.CUDAForm_float64(self._ctx, cpp_form, ufcx_form_addr)
        else:
            raise ValueError(f"Cannot instantiate CUDAForm for Form of type {type(cpp_form)}!")

        # TODO expose these parameters to the user
        self._cuda_form.compile(self._ctx, max_threads_per_block=1024, min_blocks_per_multiprocessor=1)

    @property
    def cuda_form():
        """Return the underlying cpp CUDAForm
        """

        return self._cuda_form

    @property
    def cuda_mesh():
        """Return the underlying cpp CUDAMesh"""

        return self._cuda_mesh

    @property
    def dolfinx_form():
        """Return the underlying Dolfinx form
        """

        return self._dolfinx_form

def form(form: ufl.Form, **kwargs):
    """Create a CUDAForm from a ufl form
    """

    form_compiler_options = kwargs.get("form_compiler_options", dict())
    form_compiler_options["cuda"] = True
    dolfinx_form = fe.form(form)
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

