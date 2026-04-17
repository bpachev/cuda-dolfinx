# Copyright (C) 2026 Jonas Heinzmann, Benjamin Pachev
#
# This file is part of cuDOLFINX
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import argparse as ap
import time

import petsc4py
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import cudolfinx as cufem
import dolfinx
import ufl
from cudolfinx.petsc import NonlinearProblem as cuNonlinearProblem
from dolfinx import mesh
from dolfinx.fem.petsc import NonlinearProblem
from ufl import dx, grad, inner

petsc4py.init(comm=MPI.COMM_WORLD)


# petsc logging to see GPU utilization, CpuToGpu and GpuToCpu times, etc.
opts = PETSc.Options()
opts.setValue("log_view", ":petsc.log")
PETSc.Log.begin()


def create_mesh(res: int = 10, dim: int = 3):
    """Create mesh."""
    if dim == 3:
        return mesh.create_box(
            comm=MPI.COMM_WORLD,
            points=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            n=(res, res, res),
            cell_type=mesh.CellType.tetrahedron,
        )
    else:
        return mesh.create_unit_square(MPI.COMM_WORLD, res, res)


def main(res: int = 30, degree: int = 1, dim: int = 3, cuda: bool = True):
    """Solve a nonlinear problem on a CPU or GPU."""
    domain = create_mesh(res, dim=dim)
    comm = domain.comm

    if cuda and comm.size > 1:
        if comm.rank == 0:
            print("Using ghost layer mesh for CUDA Assembly")
        domain = cufem.ghost_layer_mesh(domain)

    V = dolfinx.fem.functionspace(domain, ("Lagrange", degree))

    if comm.rank == 0:
        dofs_global = V.dofmap.index_map.size_global
        elements_global = domain.topology.index_map(domain.topology.dim).size_global
        print(f"{elements_global} elements, {dofs_global} DOFs\n")

    u = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)

    if dim == 3:
        f = 10.0 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 + (x[2] - 0.5) ** 2) / 0.02)
    else:
        f = 10.0 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)

    F = (1.0 + u**2) * inner(grad(u), grad(v)) * dx - f * v * dx
    J = ufl.derivative(F, u)

    boundary_facets = mesh.locate_entities_boundary(
        domain,
        dim=domain.topology.dim - 1,
        marker=lambda x: np.ones(x.shape[1], dtype=bool),
    )
    dofs = dolfinx.fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_monitor": "",
        "snes_linesearch_type": "bisection",
        "snes_linesearch_damping": 1.0,
        "snes_linesearch_monitor": "",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 25,
        "snes_converged_reason": "",
        "snes_error_if_not_converged": True,
        "ksp_type": "cg",
        "ksp_rtol": 1e-12,
        "ksp_max_it": 500,
        "ksp_converged_reason": "",
        "ksp_error_if_not_converged": True,
        "pc_type": "gamg",
    }

    if cuda:
        # GPU version
        problem = cuNonlinearProblem(
            F,
            u,
            petsc_options_prefix="nl_",
            bcs=[bc],
            petsc_options=petsc_options,
            J=J,
        )
    else:
        # CPU version
        problem = NonlinearProblem(
            F,
            u,
            petsc_options_prefix="nl_",
            bcs=[bc],
            petsc_options=petsc_options,
            J=J,
        )

    t0 = time.perf_counter()
    problem.solve()
    comm.Barrier()
    elapsed = time.perf_counter() - t0

    if comm.rank == 0:
        print(f"\nelapsed time: {elapsed:.3f} s {'(GPU)' if cuda else '(CPU)'}")


if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Nonlinear problem demo using NonlinearProblem class")
    parser.add_argument(
        "--res",
        default=10,
        type=int,
        help="Number of subdivisions per dimension (default: 10)",
    )
    parser.add_argument(
        "--degree",
        default=1,
        type=int,
        help="Lagrange polynomial degree (default: 1)",
    )
    parser.add_argument(
        "--dim",
        default=3,
        type=int,
        choices=[2, 3],
        help="Spatial dimension (default: 3)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Run only the CPU version (default: False)",
    )
    args = parser.parse_args()

    main(res=args.res, degree=args.degree, dim=args.dim, cuda=not args.no_cuda)
