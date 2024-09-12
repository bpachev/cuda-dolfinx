// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <concepts>
#include <cudolfinx/la/petsc.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/la/petsc.h>
#include <map>
#include <memory>
#include <petscmat.h>
#include <petscvec.h>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx::fem
{

namespace petsc
{	

template <std::floating_point T>
Mat create_cuda_matrix(const Form<PetscScalar, T>& a)
{
  la::SparsityPattern pattern = fem::create_sparsity_pattern(a);
  pattern.finalize();
  return la::petsc::create_cuda_matrix(a.mesh()->comm(), pattern);
}

} // namespace petsc
} // namespace dolfinx::fem
