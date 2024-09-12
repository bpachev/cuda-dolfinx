// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/la/SparsityPattern.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <petscvec.h>
#include <petscmacros.h>

namespace dolfinx::la
{

namespace petsc
{

Mat create_cuda_matrix(MPI_Comm comm, const SparsityPattern& sp);

} // namespace petsc
} // namespace dolfinx::la
