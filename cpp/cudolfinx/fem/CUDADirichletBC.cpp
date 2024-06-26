// Copyright (C) 2020 James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CUDADirichletBC.h"
#include <cudolfinx/common/CUDA.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FunctionSpace.h>

#if defined(HAS_CUDA_TOOLKIT)
#include <cuda.h>
#endif

using namespace dolfinx;
using namespace dolfinx::fem;

#if defined(HAS_CUDA_TOOLKIT)
  #endif
