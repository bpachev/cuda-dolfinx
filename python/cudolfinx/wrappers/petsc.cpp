// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of cuDOLFINX
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "caster_petsc.h"
#include <cudolfinx/fem/petsc.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <petsc4py/petsc4py.h>
#include <petscis.h>

namespace
{

void petsc_fem_module(nb::module_& m)
{
  m.def("create_cuda_matrix", dolfinx::fem::petsc::create_cuda_matrix<PetscReal>,
        nb::rv_policy::take_ownership, nb::arg("a"),
        "Create a PETSc CUDA Mat for a bilinear form.");
  m.def("create_cuda_block_matrix", dolfinx::fem::petsc::create_cuda_block_matrix<PetscReal>,
        nb::rv_policy::take_ownership, nb::arg("forms"),
        "Create a monolithic PETSc CUDA Mat from a list of lists of bilinear forms.");
}

} // namespace

namespace cudolfinx_wrappers
{
void petsc(nb::module_& m_fem)
{
  nb::module_ petsc_fem_mod
      = m_fem.def_submodule("petsc", "PETSc-specific finite element module");
  petsc_fem_module(petsc_fem_mod);
}
} // namespace cudolfinx_wrappers

