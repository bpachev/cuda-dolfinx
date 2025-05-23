# Copyright (C) 2024 Benjamin Pachev
#
# This file is part of cuDOLFINX
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import argparse as ap
from mpi4py import MPI
from petsc4py import PETSc
import cudolfinx as cufem
from dolfinx import fem as fe, mesh
from dolfinx.fem import petsc as fe_petsc
import numpy as np
import ufl
import time
from ufl import dx, ds, grad, inner 

def create_mesh(res: int = 10, dim: int = 3):
    """Create a uniform tetrahedral mesh on the unit cube.

    Parameters
    ----------
    res - Number of subdivisions along each dimension
    dim - Geometric dimension of mesh

    Returns
    ----------
    mesh - The mesh object.
    """

    if dim == 3:
        return mesh.create_box(
            comm = MPI.COMM_WORLD,
            points = ((0,0,0), (1, 1, 1)),
            n = (res, res, res),
            cell_type = mesh.CellType.tetrahedron
        )
    elif dim == 2:
        return mesh.create_unit_square(MPI.COMM_WORLD, res, res)

def main(res, cuda=True, degree=1, dim=3):
    """Assembles a stiffness matrix for the Poisson problem with the given resolution.
    """

    domain = create_mesh(res, dim=dim)
    comm = domain.comm
    if cuda and comm.size > 1:
        if comm.rank == 0:
            print("Using ghost layer mesh for CUDA Assembly")
        domain = cufem.ghost_layer_mesh(domain)

    V = fe.functionspace(domain, ("Lagrange", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    if dim == 3:
        f = 10*ufl.exp(-((x[0]-.05)**2 + (x[1]-.05)**2 + (x[2]-.05)**2) / .02)
    elif dim == 2:
        f = 10*ufl.exp(-((x[0]-.05)**2 + (x[1]-.05)**2) / .02)
    g = ufl.sin(5*x[0])*ufl.sin(5*x[1])
    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx + inner(g, v) * ds

    facets = mesh.locate_entities_boundary(
      domain,
      dim=(domain.topology.dim - 1),
      marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 2.0),
    )

    dofs = fe.locate_dofs_topological(V=V, entity_dim=domain.topology.dim-1, entities=facets)
    bc = fe.dirichletbc(value=PETSc.ScalarType(0), dofs=dofs, V=V)

    if cuda:
        a = cufem.form(a)
        L = cufem.form(L)
        asm = cufem.CUDAAssembler()
        A = asm.create_matrix(a)
        b = asm.create_vector(L)
        device_bcs = asm.pack_bcs([bc])
    else:
        a = fe.form(a, jit_options = {"cffi_extra_compile_args":["-O3", "-mcpu=neoverse-v2"]})
        L = fe.form(L, jit_options = {"cffi_extra_compile_args":["-O3", "-mcpu=neoverse-v2"]})
        A = fe_petsc.create_matrix(a)
        b = fe_petsc.create_vector(L)
    start = time.time()
    if cuda:
        asm.assemble_matrix(a, A, bcs=device_bcs)
        A.assemble()
    else:
        fe_petsc.assemble_matrix(A, a, bcs=[bc])
        A.assemble()
    elapsed = time.time()-start

    timing = comm.gather(elapsed, root=0)
    if comm.rank == 0:
        timing = np.asarray(timing)
        timing = np.max(timing)
        # show max over all MPI processes, as that's the rate-limiter
        print(f"Res={res}, Num cells", domain.topology.index_map(domain.topology.dim).size_global)
        print(f"Assembly timing: {timing}, Dofs: {V.dofmap.index_map.size_global}")

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--res", default=10, type=int, help="Number of subdivisions in each dimension.")
    parser.add_argument("--degree", default=1, type=int, help="Polynomial degree.")
    parser.add_argument("--dim", default=3, type=int, help="Geometric dimension.")
    parser.add_argument("--no-cuda", default=False, action="store_true", help="Disable GPU acceleration.")
    args = parser.parse_args()

    main(res=args.res, cuda = not args.no_cuda, degree=args.degree, dim=args.dim)
