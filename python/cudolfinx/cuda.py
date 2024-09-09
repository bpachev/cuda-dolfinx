from __future__ import annotations

import collections
import functools
import typing
import tempfile

import dolfinx
from dolfinx import cpp as _cpp
from dolfinx import la
from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.forms import Form
from dolfinx.fem.function import Function, FunctionSpace
from dolfinx.mesh import Mesh
from dolfinx import fem as fe
from cudolfinx import cpp as _cucpp
from petsc4py import PETSc
import gc
import numpy as np

def init_device():
  """Initialize PETSc device
  """

  d = PETSc.Device()
  d.create(PETSc.Device.Type.CUDA)
  return d

def create_petsc_cuda_vector(L: Form) -> PETSc.Vec:
  """Create PETSc Vector on device
  """

  index_map = L.function_spaces[0].dofmap.index_map
  bs = L.function_spaces[0].dofmap.index_map_bs
  size = (index_map.size_local * bs, index_map.size_global * bs)
  # we need to provide at least the local CPU array
  arr = np.zeros(size[0])
  return PETSc.Vec().createCUDAWithArrays(cpuarray=arr, size=size, bsize=bs, comm=index_map.comm)

def _create_form_on_device(form: Form, ctx: _cucpp.fem.CUDAContext):
  """Create device-side data structures needed to assemble a Form
  """

  # prevent duplicate initialization of CUDA data
  if hasattr(form, '_cuda_form'): return

  # now determine the Mesh object corresponding to this form
  form._cuda_mesh = _create_mesh_on_device(form.mesh, ctx)

  cpp_form = form._cpp_object
  if type(cpp_form) is _cpp.fem.Form_float32:
    cuda_form = _cucpp.fem.CUDAForm_float32(ctx, cpp_form)
  elif type(cpp_form) is _cpp.fem.Form_float64:
    cuda_form = _cucpp.fem.CUDAForm_float64(ctx, cpp_form)
  else:
    raise ValueError(f"Cannot instantiate CUDAForm for Form of type {type(cpp_form)}!")

  # TODO expose these to the user. . . .
  cuda_form.compile(ctx, max_threads_per_block=1024, min_blocks_per_multiprocessor=1)
  form._cuda_form = cuda_form

def _create_mesh_on_device(cpp_mesh: typing.Union[_cpp.mesh.Mesh_float32, _cpp.mesh.Mesh_float64], ctx: _cucpp.fem.CUDAContext):
  """Create device-side mesh data
  """

  if type(cpp_mesh) is _cpp.mesh.Mesh_float32:
    return _cucpp.fem.CUDAMesh_float32(ctx, cpp_mesh)
  elif type(cpp_mesh) is _cpp.mesh.Mesh_float64:
    return _cucpp.fem.CUDAMesh_float64(ctx, cpp_mesh)
  else:
    raise ValueError(f"Cannot instantiate CUDAMesh for Mesh of type {type(cpp_mesh)}!")



@functools.singledispatch
def to_device(obj: typing.Any, ctx: _cucpp.fem.CUDAContext):
  """Copy an object to the device
  """

  return _to_device(obj, ctx)

@to_device.register(Form)
def _to_device(a: Form, ctx: _cucpp.fem.CUDAContext):
  """Copy a Form's coefficients to the device
  """

  _create_form_on_device(a, ctx)
  a._cuda_form.to_device(ctx)

@to_device.register(Function)
def _to_device(f: Function, ctx: _cucpp.fem.CUDAContext):
  """Copy a Function to the device
  """

  f._cpp_object.x.to_device(ctx)

@to_device.register(la.Vector)
def _to_device(v: la.Vector, ctx: _cucpp.fem.CUDAContext):
  """Copy a Vector to the device
  """

  v._cpp_object.to_device(ctx)

class CUDAAssembler:
  """Class for assembly on the GPU
  """

  def __init__(self):
    """Initialize the assembler
    """

    self._device = init_device()
    self._ctx = _cucpp.fem.CUDAContext()
    self._tmpdir = tempfile.TemporaryDirectory()
    self._cpp_object = _cucpp.fem.CUDAAssembler(self._ctx, self._tmpdir.name)

  def assemble_matrix(self,
      a: Form,
      mat: typing.Optional[_cucpp.fem.CUDAMatrix] = None,
      bcs: typing.Optional[typing.Union[list[DirichletBC], CUDADirichletBC]] = None,
      diagonal: float = 1.0,
      constants: typing.Optional[list] = None,
      coeffs: typing.Optional[list] = None
  ):
    """Assemble bilinear form into a matrix on the GPU.

    Args:
        a: The bilinear form assemble.
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

    self._check_form(a)

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

    _bc0 = bc_collection._get_cpp_bcs(a.function_spaces[0])
    _bc1 = bc_collection._get_cpp_bcs(a.function_spaces[1])

    self.pack_coefficients(a, coeffs)

    _cucpp.fem.assemble_matrix_on_device(
       self._ctx, self._cpp_object, a._cuda_form,
       a._cuda_mesh, mat._cpp_object, _bc0, _bc1
    )

    return mat

  def assemble_vector(self,
    b: Form,
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

    self._check_form(b)
    if vec is None:
      vec = self.create_vector(b)
   
    self.pack_coefficients(b, coeffs) 
    _cucpp.fem.assemble_vector_on_device(self._ctx, self._cpp_object, b._cuda_form,
      b._cuda_mesh, vec._cpp_object)
    return vec

  def create_matrix(self, a: Form) -> CUDAMatrix:
    """Create a CUDAMatrix from a given form
    """
    petsc_mat = _cucpp.fem.petsc.create_cuda_matrix(a._cpp_object)
    return CUDAMatrix(self._ctx, petsc_mat)

  def create_vector(self, b: Form) -> CUDAVector:
    """Create a CUDAVector from a given form
    """

    petsc_vec = create_petsc_cuda_vector(b)
    return CUDAVector(self._ctx, petsc_vec)

  def pack_bcs(self, bcs: list[DirichletBC]) -> CUDADirichletBC:
    """Pack boundary conditions into a single object for use in assembly.

    The returned object is of type CUDADirichletBC and can be used in place of a list of
    regular DirichletBCs. This is more efficient when performing multiple operations with the same list of 
    boundary conditions, or when boundary condition values need to change over time.
    """

    return CUDADirichletBC(self._ctx, bcs)

  def pack_coefficients(self, a: Form, coefficients: typing.Optional[list[Function]]=None):
    """Pack coefficients on device
    """
  
    self._check_form(a)
    if coefficients is None:
      _cucpp.fem.pack_coefficients(self._ctx, self._cpp_object, a._cuda_form)
    else:
      # perform a repacking with only the indicated coefficients
      _coefficients = [c._cpp_object for c in coefficients]
      _cucpp.fem.pack_coefficients(self._ctx, self._cpp_object, a._cuda_form, _coefficients)

  def apply_lifting(self,
    b: CUDAVector,
    a: list[Form],
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
      self._check_form(form)
      self.pack_coefficients(form, form_coeffs)
      cuda_forms.append(form._cuda_form)
      if cuda_mesh is None: cuda_mesh = form._cuda_mesh
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

  def to_device(self, obj: typing.Any):
    """Copy an object needed for assembly to the device
    """

    to_device(obj, self._ctx)

  def _check_form(self, a: Form):
    """Check the provided Form to ensure it has been created on the device
    """

    if not hasattr(a, '_cuda_form'):
      self.to_device(a)
    
class CUDAVector:
  """Vector on device
  """

  def __init__(self, ctx, vec):
    """Initialize the vector
    """

    self._petsc_vec = vec
    self._ctx = ctx
    self._cpp_object = _cucpp.fem.CUDAVector(ctx, self._petsc_vec)

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

  def mat(self):
    """Return underlying CUDA matrix
    """

    return self._petsc_mat

  def to_host(self):
    """Copy device-side values to host
    """

    self._cpp_object.to_host(self._ctx)

  def __del__(self):
    """Delete the matrix and free up GPU resources
    """

    # make sure we delete the CUDAMatrix before the petsc matrix
    del self._cpp_object
   

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

  def update(self, bcs: List[DirichletBC]):
    """Update a subset of the boundary conditions.

    Used for cases with time-varying boundary conditions whose device-side values
    need to be updated.
    """

    _bcs_to_update = [bc._cpp_object for bc in bcs]
    for _cpp_bc, V in zip(self._cpp_bc_objects, self._function_spaces):
      # filter out anything not contained in the right function space
      _cpp_bc.update([_bc for _bc in _bcs_to_update if V.contains(_bc.function_space)])

