from petsc4py import PETSc
from cudolfinx import cpp as _cucpp

_device = None

def _init_device():
  """Initialize PETSc device
  """
  global _device  
  d = PETSc.Device()
  d.create(PETSc.Device.Type.CUDA)
  _device = d

def get_device():
  """Return PETSc device
  """

  global _device
  if _device is None:
    _init_device()
  return _device

def get_cuda_context():
  """Return the CUDA context, intializing it if needed
  """
  global _device
  if _device is None:
     _init_device() 
  return _cucpp.fem.CUDAContext()
