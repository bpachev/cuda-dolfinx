// Copyright (C) 2020 James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CUDAMeshEntities.h"
#include <cudolfinx/common/CUDA.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/graph/AdjacencyList.h>

#if defined(HAS_CUDA_TOOLKIT)
#include <cuda.h>
#endif

using namespace dolfinx;
using namespace dolfinx::mesh;

#if defined(HAS_CUDA_TOOLKIT)
  
#endif
