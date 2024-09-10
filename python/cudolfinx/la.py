from cudolfinx import cpp as _cucpp

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
   

