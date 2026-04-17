# Copyright (C) 2026 Jonas Heinzmann, Benjamin Pachev
#
# This file is part of cuDOLFINX
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""High-level solver classes."""

from __future__ import annotations

from collections.abc import Sequence

from petsc4py import PETSc

import ufl
from cudolfinx.assemble import CUDAAssembler
from cudolfinx.form import CUDAForm
from cudolfinx.form import form as _cuda_form
from cudolfinx.la import CUDAMatrix, CUDAVector
from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.forms import Form
from dolfinx.fem.function import Function


class NonlinearProblem:
    """High-level class for solving nonlinear variational problems with PETSc SNES on the GPU.

    It is adapted from and and resembles the interface of dolfinx.fem.petsc.NonlinearProblem.
    (currently not supporting block/multi-form problems)
    """

    def __init__(
        self,
        F: ufl.Form,
        u: Function,
        *,
        petsc_options_prefix: str,
        bcs: Sequence[DirichletBC] | None = None,
        J: ufl.Form | None = None,
        P: ufl.Form | None = None,
        petsc_options: dict | None = None,
        cuda_jit_options: dict | None = None,
    ):
        """Initialise the GPU nonlinear problem.

        Args:
        F: UFL form(s) representing the residual
        u: Current-iterate function(s). Must be the same object(s)
            used as coefficients in ``F`` and ``J``
        petsc_options_prefix: Mandatory options prefix used as root
            prefix on all internally created PETSc objects
        bcs: Dirichlet boundary conditions
        J: UFL form(s) for the Jacobian. Derived automatically via
            ``ufl.derivative`` when not provided (single-form only)
        P: UFL form(s) for an optional preconditioner matrix
        petsc_options: Options forwarded to the PETSc SNES solver
        cuda_jit_options: Passed to the CUDA JIT compiler
        """
        # check types of forms
        assert isinstance(F, ufl.Form), (
            f"F must be a ufl.Form, got {type(F).__name__}. "
            "Block/multi-form problems are not yet supported."
        )
        if J is not None:
            assert isinstance(J, ufl.Form), (
                f"J must be a ufl.Form, got {type(J).__name__}. "
                "Block/multi-form problems are not yet supported."
            )
        if P is not None:
            assert isinstance(P, ufl.Form), (
                f"P must be a ufl.Form, got {type(P).__name__}. "
                "Block/multi-form problems are not yet supported."
            )

        if petsc_options_prefix == "":
            raise ValueError("PETSc options prefix cannot be empty.")

        self._assembler = CUDAAssembler()
        self._u = u
        bcs = [] if bcs is None else list(bcs)
        self._bcs = bcs
        self._cuda_bcs = self._assembler.pack_bcs(bcs)

        # automatically derive Jacobian form if not provided
        if J is None:
            J = ufl.derivative(F, u, ufl.TrialFunction(u.function_space))

        # instantiate CUDA forms
        self._cuda_F: CUDAForm = _cuda_form(F, cuda_jit_args=cuda_jit_options or {})
        self._cuda_J: CUDAForm = _cuda_form(J, cuda_jit_args=cuda_jit_options or {})
        self._cuda_P: CUDAForm | None = (
            _cuda_form(P, cuda_jit_args=cuda_jit_options or {}) if P is not None else None
        )

        # Jacobian matrix
        self._cuda_A = self._assembler.create_matrix(self._cuda_J)

        # preconditioner matrix
        if self._cuda_P is not None:
            self._cuda_P_mat = self._assembler.create_matrix(self._cuda_P)
        else:
            self._cuda_P_mat = None

        # residual vector
        self._cuda_b = self._assembler.create_vector(self._cuda_F)

        # solution work vector for SNES (plain PETSc Vec with same layout as b)
        self._x: PETSc.Vec = self._cuda_b.vector.copy()  # type: ignore[attr-defined]

        # persistent CUDA vector wrapping u (owned + ghost DOFs) for residual assembly
        self._cuda_u = CUDAVector(self._assembler._ctx, u.x.petsc_vec, include_ghosts=True)

        # SNES object
        self._snes: PETSc.SNES = PETSc.SNES().create(  # type: ignore[attr-defined]
            u.function_space.mesh.comm
        )

        self._snes.setFunction(self._assemble_residual, self._cuda_b.vector)
        self._snes.setJacobian(
            self._assemble_jacobian,
            self._cuda_A.mat,
            self._cuda_P_mat.mat if self._cuda_P_mat is not None else None,
        )

        # options prefixes
        self._snes.setOptionsPrefix(petsc_options_prefix)
        self._cuda_A.mat.setOptionsPrefix(f"{petsc_options_prefix}A_")
        if self._cuda_P_mat is not None:
            self._cuda_P_mat.mat.setOptionsPrefix(f"{petsc_options_prefix}P_mat_")
        self._cuda_b.vector.setOptionsPrefix(f"{petsc_options_prefix}b_")
        self._x.setOptionsPrefix(f"{petsc_options_prefix}x_")

        # forward SNES options
        if petsc_options is not None:
            opts = PETSc.Options()  # type: ignore[attr-defined]
            opts.prefixPush(self._snes.getOptionsPrefix())
            for k, v in petsc_options.items():
                opts[k] = v
            self._snes.setFromOptions()
            for k in petsc_options:
                del opts[k]
            opts.prefixPop()

    def _assemble_residual(
        self,
        snes: PETSc.SNES,  # type: ignore[name-defined]
        x: PETSc.Vec,  # type: ignore[name-defined]
        b: PETSc.Vec,  # type: ignore[name-defined]
    ) -> None:
        """Assemble the residual ``F(u)`` on the GPU.

        This is called by PETSc SNES on every function evaluation.

        Note: For the line searches, PETSc has an internal work vector
        also for the residual, which it passes as the ``b`` argument to this function.
        This is the vector that must be updated with the assembled residual,
        and not the original residual vector created in ``__init__``
        """
        # copy x to u and update ghosts
        x.copy(self._u.x.petsc_vec)
        self._u.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        # update the persistent CUDA vector with the current u values (including ghosts)
        self._cuda_u.to_device()

        # wrap b vector passed in from SNES and assemble directly into it, avoiding a copy
        cuda_b = CUDAVector(self._assembler._ctx, b)

        # assemble and apply BCs
        self._assembler.assemble_vector(self._cuda_F, cuda_b, zero=True)
        self._assembler.apply_lifting(
            cuda_b,
            [self._cuda_J],
            [self._cuda_bcs],
            x0=[self._cuda_u],
            scale=-1.0,
        )
        self._assembler.set_bc(
            cuda_b,
            bcs=self._cuda_bcs,
            V=self._u.function_space,
            x0=self._cuda_u,
            scale=-1.0,
        )

    def _assemble_jacobian(
        self,
        snes: PETSc.SNES,  # type: ignore[name-defined]
        x: PETSc.Vec,  # type: ignore[name-defined]
        A: PETSc.Mat,  # type: ignore[name-defined]
        P: PETSc.Mat,  # type: ignore[name-defined]
    ) -> None:
        """Assemble the Jacobian (and optional preconditioner) on the GPU.

        This is called by PETSc SNES on every Jacobian evaluation.
        """
        # copy x to u
        x.copy(self._u.x.petsc_vec)
        self._u.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        # assemble Jacobian
        self._assembler.assemble_matrix(self._cuda_J, self._cuda_A, bcs=self._cuda_bcs)
        self._cuda_A.assemble()
        A.assemble()

        # assemble preconditioner
        if self._cuda_P_mat is not None and self._cuda_P is not None:
            self._assembler.assemble_matrix(self._cuda_P, self._cuda_P_mat, bcs=self._cuda_bcs)
            self._cuda_P_mat.assemble()
            P.assemble()

    def solve(self) -> Function:
        """Solve the nonlinear problem with PETSc SNES.

        Note:
        It is the caller's responsibility to check convergence, either with
        assert problem.solver.getConvergedReason() > 0
        or by configuring SNES options to raise an error on non-convergence
        petsc_options = {"snes_error_if_not_converged": True}

        For the line searches, PETSc has internal work vectors for the solution,
        which it passes as the ``x`` argument to the SNES functions.
        This is the vector that is updated with the solution, and must not
        share memory with self._u at which the residual, and Jacobian
        (and preconditioner) are defined. Hence, the solution must be copied
        from self._u to the SNES work vector before solving, and back to self._u
        after solving

        Returns:
        The updated solution function(s) ``u``
        """
        # copy u to x
        self._u.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        self._u.x.petsc_vec.copy(self._x)

        # update BCs
        self._cuda_bcs.update(self._bcs)

        # solve
        self._snes.solve(None, self._x)

        # copy x to u
        self._x.copy(self._u.x.petsc_vec)
        self._u.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        return self._u

    def __del__(self) -> None:
        """Destroy PETSc objects created internally."""
        for obj in filter(
            None,
            (
                self._snes,
                self._cuda_A.mat if self._cuda_A is not None else None,
                self._cuda_P_mat.mat if self._cuda_P_mat is not None else None,
                self._cuda_b.vector if self._cuda_b is not None else None,
                self._x,
            ),
        ):
            try:
                obj.destroy()
            except Exception:
                pass

    @property
    def F(self) -> Form:
        """Compiled dolfinx residual form."""
        return self._cuda_F.dolfinx_form

    @property
    def J(self) -> Form:
        """Compiled dolfinx Jacobian form."""
        return self._cuda_J.dolfinx_form

    @property
    def preconditioner(self) -> Form:
        """Compiled dolfinx preconditioner form, or ``None``."""
        if self._cuda_P is None:
            return None

        return self._cuda_P.dolfinx_form

    @property
    def cuda_F(self) -> CUDAForm:
        """GPU residual form."""
        return self._cuda_F

    @property
    def cuda_J(self) -> CUDAForm:
        """GPU Jacobian form."""
        return self._cuda_J

    @property
    def cuda_A(self) -> CUDAMatrix:
        """GPU Jacobian matrix."""
        return self._cuda_A

    @property
    def cuda_b(self) -> CUDAVector:
        """GPU residual vector."""
        return self._cuda_b

    @property
    def A(self) -> PETSc.Mat:  # type: ignore[name-defined]
        """Jacobian matrix (host-side PETSc Mat)."""
        return self._cuda_A.mat

    @property
    def P_mat(self) -> PETSc.Mat | None:  # type: ignore[name-defined]
        """Preconditioner matrix (host-side PETSc Mat), or ``None``."""
        return self._cuda_P_mat.mat if self._cuda_P_mat is not None else None

    @property
    def b(self) -> PETSc.Vec:  # type: ignore[name-defined]
        """Residual vector (host-side PETSc Vec)."""
        return self._cuda_b.vector

    @property
    def x(self) -> PETSc.Vec:  # type: ignore[name-defined]
        """SNES solution work vector."""
        return self._x

    @property
    def solver(self) -> PETSc.SNES:  # type: ignore[name-defined]
        """The underlying PETSc SNES solver."""
        return self._snes

    @property
    def u(self) -> Function:
        """Solution function."""
        return self._u
