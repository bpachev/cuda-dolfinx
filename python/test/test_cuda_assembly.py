# Copyright (C) 2024 Benjamin Pachev
#
# This file is part of cuDOLFINX
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import petsc4py
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem as fe, mesh
from dolfinx.fem import petsc
import ufl
import numpy as np
import cudolfinx as cufem
from basix.ufl import element, mixed_element

"""
@author Benjamin Pachev <benjamin.pachev@gmail.com>
@copyright 2024

A set of simple variational forms to test the correctness of CUDA-accelerated assembly.
"""


def make_mixed_form():
  """Test compilation of a mixed form.
  """

  domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10, mesh.CellType.triangle)
  el = element("P", domain.basix_cell(), 1)
  
  V = fe.functionspace(domain, el)
  u = ufl.TrialFunction(V)
  p = ufl.TestFunction(V)
  A = ufl.dot(ufl.grad(u), ufl.grad(p)) * ufl.dx
  F = fe.form(A)
  mat = fe.assemble_matrix(F)

def make_test_domain():
  """Make a test domain
  """

  n = 19
  m = 27
  return mesh.create_unit_square(MPI.COMM_WORLD, n, m, mesh.CellType.triangle)

def make_ufl(domain=None):
  """Create the UFL needed for making the forms
  """

  if domain is None:
    domain = make_test_domain()
  
  V = fe.functionspace(domain, ("P", 1))
  V_dg = fe.functionspace(domain, ("DG", 1))
  u = fe.Function(V)
  p = ufl.TestFunction(V)
  p_dg = ufl.TestFunction(V_dg)
  n = ufl.FacetNormal(domain)
  u.interpolate(lambda x: x[0]**2 + x[1])
  u_dg = fe.Function(V_dg)
  u_dg.interpolate(lambda x: x[0]**2 + x[1])
  kappa = fe.Function(V)
  kappa.interpolate(lambda x: np.sin(x[0])*np.cos(x[1]))

  cell_residual = (ufl.exp(u)*p*kappa + ufl.dot(ufl.grad(u), ufl.grad(p))) * ufl.dx
  exterior_facet_residual = u*kappa*p * ufl.dot(ufl.grad(u), n) * ufl.ds
  interior_facet_residual = ufl.avg(p_dg) * ufl.avg(kappa) * ufl.dot(ufl.avg(ufl.grad(u_dg)), n("+")) * ufl.dS
  #interior_facet_residual = ufl.avg(p_dg) * ufl.avg(kappa) * ufl.dot(ufl.avg(ufl.grad(u)), n("+")) * ufl.dS

  cell_jac = ufl.derivative(cell_residual, u)
  exterior_jac = ufl.derivative(exterior_facet_residual, u)
  #interior_jac = ufl.derivative(interior_facet_residual, u)
  interior_jac = ufl.derivative(interior_facet_residual, u_dg)

  f = fe.Function(V)
  f.interpolate(lambda x: x[0] +x[1])
  dofs = fe.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
  bc = fe.dirichletbc(f, dofs)

  return {
           "coeff": kappa,
           "bcs": [bc],
           "vector": [cell_residual, exterior_facet_residual, interior_facet_residual],
           "matrix": [cell_jac, exterior_jac, interior_jac]}

def test_assembly():
  """Test correctness of assembly
  """

  ufl_forms = make_ufl()

  for i, form in enumerate(ufl_forms["vector"]):
      fenics_form = fe.form(form)
      vec = petsc.create_vector(fenics_form)
      petsc.assemble_vector(vec, fenics_form)

  for i, form in enumerate(ufl_forms["matrix"]):
      fenics_form = fe.form(form)
      mat = petsc.create_matrix(fenics_form)
      mat.zeroEntries()
      petsc.assemble_matrix(mat, fenics_form)
      mat.assemble()

def compare_mats(matcsr, matpetsc):
  """Compare a native FEniCS MatrixCSR to a PETSc matrix
  """

  indptr, indices, data = matpetsc.getValuesCSR()
  bad = np.where(~np.isclose(matcsr.data, data))[0]
  assert np.allclose(matcsr.data, data)

def compare_vecs(vecfenics, vecpetsc):
  assert np.allclose(vecfenics.array, vecpetsc.array)

def test_cuda_assembly():
  """Check assembly on GPU
  """


  ufl_forms = make_ufl()
  asm = cufem.CUDAAssembler()

  for i, form in enumerate(ufl_forms['vector']):
    if i == 0: continue
    f = fe.form(form)
    vec1 = fe.assemble_vector(f)
    vec2 = asm.assemble_vector(cufem.form(form))
    compare_vecs(vec1, vec2.vector)

  for i, form in enumerate(ufl_forms['matrix']):
    f = fe.form(form)
    Mat1 = fe.assemble_matrix(f, bcs=ufl_forms['bcs'])
    Mat2 = asm.assemble_matrix(cufem.form(form), bcs=ufl_forms['bcs'])
    Mat2.assemble()
    # now we need to compare the two
    compare_mats(Mat1, Mat2.mat)

def test_reassembly():
  """Ensure correct assembly when coefficients are updated
  """

  ufl_forms = make_ufl()
  coeff = ufl_forms["coeff"]
  cuda_vec_form = cufem.form(ufl_forms["vector"][0])
  vec_form = cuda_vec_form.dolfinx_form
  #mat_form = fe.form(ufl_forms["matrix"][0])
  asm = cufem.CUDAAssembler()
  vec_cuda = asm.assemble_vector(cuda_vec_form)
  vec_fe = fe.assemble_vector(vec_form)
  compare_vecs(vec_fe, vec_cuda.vector)

  for d in [2,3]:
    coeff.interpolate(lambda x: x[0]**d + x[1]**d)
    vec_fe.array[:] = 0
    cuda_vec_form.to_device()
    fe.assemble_vector(vec_fe.array, vec_form)
    asm.assemble_vector(cuda_vec_form, vec_cuda)

    compare_vecs(vec_fe, vec_cuda.vector)

def test_lifting():
  """Ensure lifting and bc setting work correctly
  """

  ufl_forms = make_ufl()
  asm = cufem.CUDAAssembler()
  for vec_form, mat_form in zip(ufl_forms['vector'][1:2], ufl_forms['matrix'][1:2]):
    L = fe.form(vec_form)
    vec_cuda = asm.assemble_vector(cufem.form(vec_form))
    vec_fe = fe.assemble_vector(L)
    cuda_a = cufem.form(mat_form)
    a = cuda_a.dolfinx_form
    compare_vecs(vec_fe, vec_cuda.vector)
    fe.set_bc(vec_fe.array, ufl_forms['bcs'])
    asm.set_bc(vec_cuda, ufl_forms['bcs'], L.function_spaces[0])
    compare_vecs(vec_fe, vec_cuda.vector)
    fe.apply_lifting(vec_fe.array, [a], [ufl_forms['bcs']])
    asm.apply_lifting(vec_cuda, [cuda_a], [ufl_forms['bcs']])
    compare_vecs(vec_fe, vec_cuda.vector)

