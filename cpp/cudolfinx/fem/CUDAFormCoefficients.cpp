// Copyright (C) 2020 James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CUDAFormCoefficients.h"
#include <cudolfinx/common/CUDA.h>
#include <cudolfinx/fem/CUDADofMap.h>
#include <cudolfinx/la/CUDAVector.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/common/IndexMap.h>


#if defined(HAS_CUDA_TOOLKIT)
#include <cuda.h>
#endif

using namespace dolfinx;
using namespace dolfinx::fem;

#if defined(HAS_CUDA_TOOLKIT)

//-----------------------------------------------------------------------------
#endif
