#!/usr/bin/env python3

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import cudolfinx as cufem
import time

domain = mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0,0,0]), np.array([1, 1, 1])],
    [20, 20, 20],
    cell_type=mesh.CellType.hexahedron
)

V      = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
V_from = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))

u      = fem.Function(V)
u_true = fem.Function(V)
u_from = fem.Function(V_from)


if __name__ == "__main__":
    u_from.interpolate(lambda x: [1 + 0.1*x[0]**2,
                                  1 + 0.2*x[1]**2,
                                  1 + 0.3*x[2]**2])

    niter = 5

    u_true.interpolate(u_from)
    start = time.perf_counter()
    for _ in range(niter):
        u_true.interpolate(u_from)
    
    end = time.perf_counter()
    print(f"CPU Time: {1e3*(end-start)/niter:.2f} ms")

    ##### GPU code #######################################
    coeff      = cufem.Coefficient(u)
    coeff_from = cufem.Coefficient(u_from)
    coeff.interpolate(coeff_from) # initialize

    start = time.perf_counter()
    for _ in range(niter):
        coeff.interpolate(coeff_from)

    end = time.perf_counter()
    print(f"GPU Time: {1e3*(end-start)/niter:.2f} ms")

    assert np.allclose(u_true.x.array, coeff.values(), rtol=1e-13)
    print("PASSED")
