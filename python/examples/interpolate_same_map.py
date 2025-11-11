#!/usr/bin/env python3

import argparse as ap
from mpi4py import MPI
from petsc4py import PETSc
import cudolfinx as cufem
from dolfinx import fem, mesh
from dolfinx.fem import petsc as fe_petsc
import basix.ufl
import basix
import numpy as np
import time


domain = mesh.create_unit_cube(MPI.COMM_WORLD, 20, 20, 20, mesh.CellType.tetrahedron)
element_to = basix.ufl.element("Lagrange", "tetrahedron", 4)
V = fem.functionspace(domain, element_to)
elem_from = basix.ufl.element("Lagrange", "tetrahedron", 5, basix.LagrangeVariant(12))
V_from = fem.functionspace(domain, elem_from)

u = fem.Function(V)
u_true = fem.Function(V)
u_from = fem.Function(V_from)

u_from.interpolate(lambda x: 1 + 0.1*x[0]**2 + 0.2*x[1]**2 + 0.3*x[2]**2)

if __name__ == "__main__":
    niter = 5

    u_true.interpolate(u_from)
    start = time.perf_counter()
    for _ in range(niter):
        u_true.interpolate(u_from)
    
    end = time.perf_counter()
    print(f"CPU Time: {1e3*(end-start)/niter:.2f} ms")

    ##### GPU code #######################################
    coeff = cufem.Coefficient(u)
    coeff_from = cufem.Coefficient(u_from)
    coeff.interpolate(coeff_from) # initialize

    start = time.perf_counter()
    for _ in range(niter):
        coeff.interpolate(coeff_from)

    end = time.perf_counter()
    print(f"GPU Time: {1e3*(end-start)/niter:.2f} ms")

    assert np.allclose(u_true.x.array, coeff.values(), rtol=1e-12)
    print("PASSED")
