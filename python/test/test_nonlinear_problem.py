# Copyright (C) 2026 Jonas Heinzmann, Benjamin Pachev
#
# This file is part of cuDOLFINX
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import cudolfinx as cufem
import ufl
from cudolfinx.petsc import NonlinearProblem as cuNonlinearProblem
from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from ufl import dx, grad, inner

# resolution
RES = 50

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


def _make_problem(domain):
    """Create solution function, residual form, and BCs for a nonlinear problem
    (1 + u^2) * inner(grad(u), grad(v)) * dx = f * v * dx  with u = 0 on boundary.
    """
    V = fem.functionspace(domain, ("Lagrange", 1))
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    f = 10.0 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
    F = (1.0 + u**2) * inner(grad(u + u**2), grad(v)) * dx - f * v * dx

    boundary_facets = mesh.locate_entities_boundary(
        domain, domain.topology.dim - 1, marker=lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)

    return u, F, [bc]


def _solve_cpu():
    """Solve the nonlinear problem on the CPU with dolfinx NonlinearProblem."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, RES, RES)
    u, F, bcs = _make_problem(domain)

    cpu_problem = NonlinearProblem(
        F,
        u,
        petsc_options_prefix="nl_",
        bcs=bcs,
        petsc_options=petsc_options,
    )

    if MPI.COMM_WORLD.rank == 0:
        print("_"*75, "\nSolving on CPU\n", flush=True)

    cpu_u = cpu_problem.solve()
    cpu_iters, cpu_funcevals = (
        cpu_problem.solver.getIterationNumber(),
        cpu_problem.solver.getFunctionEvaluations(),
    )

    return cpu_u, cpu_iters, cpu_funcevals


def _solve_gpu():
    """Solve the nonlinear problem on the CPU with cuda-dolfinx NonlinearProblem."""
    domain = mesh.create_unit_square(MPI.COMM_WORLD, RES, RES)

    if MPI.COMM_WORLD.size > 1:
        domain = cufem.ghost_layer_mesh(domain)

    u, F, bcs = _make_problem(domain)

    gpu_problem = cuNonlinearProblem(
        F,
        u,
        petsc_options_prefix="nl_",
        bcs=bcs,
        petsc_options=petsc_options,
    )

    if MPI.COMM_WORLD.rank == 0:
        print("_"*75, "\nSolving on GPU\n", flush=True)

    gpu_u = gpu_problem.solve()
    gpu_iters, gpu_funcevals = (
        gpu_problem.solver.getIterationNumber(),
        gpu_problem.solver.getFunctionEvaluations(),
    )

    return gpu_u, gpu_iters, gpu_funcevals


def test_nonlinear_problem():
    """CPU and GPU (single GPU, same mesh) must give the same solution."""
    cpu_u, cpu_iters, cpu_funcevals = _solve_cpu()
    gpu_u, gpu_iters, gpu_funcevals = _solve_gpu()

    if MPI.COMM_WORLD.rank == 0:
        print("\n", "_"*75)

    assert cpu_iters == gpu_iters, (
        "Number of iterations differs between CPU and GPU"
    )
    assert cpu_funcevals == gpu_funcevals, (
        "Number of function evaluations differs between CPU and GPU"
    )

    for norm, label in zip(
        [PETSc.NormType.NORM_1, PETSc.NormType.NORM_2, PETSc.NormType.NORM_INFINITY],
        ["L1", "L2", "Linf"]
    ):
        cpu_norm = cpu_u.x.petsc_vec.norm(norm)
        gpu_norm = gpu_u.x.petsc_vec.norm(norm)
        if MPI.COMM_WORLD.rank == 0:
            print(f"CPU: {cpu_norm:.16e}\t GPU: {gpu_norm:.16e} ({label} norm)")

        assert np.isclose(cpu_norm, gpu_norm), (
            f"{label} norm of solutions differs between CPU and GPU"
        )


if __name__ == "__main__":
    test_nonlinear_problem()
