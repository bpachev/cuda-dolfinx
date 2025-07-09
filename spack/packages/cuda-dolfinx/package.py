# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.package import *


class CudaDolfinx(CMakePackage):
    """CUDA accelerated extension of DOLFINx from the FEniCS project."""

    homepage = "https://github.com/bpachev/cuda-dolfinx"
    git = "https://github.com/bpachev/cuda-dolfinx.git"
    url = "https://github.com/bpachev/cuda-dolfinx/archive/refs/tags/v0.9.0.zip"

    maintainers("bpachev")
    license("LGPL-3.0-or-later", checked_by="bpachev")

    version("main", branch="main")
    version("0.9.0", sha256="5c93155e58eee139985e9e9341cf7d8b29f8c9cbc51ccdf05134cdfb70ae105d")

    depends_on("cxx", type="build")

    depends_on("fenics-dolfinx@0.9+petsc+adios2", when="@0.9:")
    depends_on("py-fenics-dolfinx@0.9", when="@0.9:")
    depends_on("petsc+shared+mpi+cuda")

    root_cmakelists_dir = "cpp"

    def cmake_args(self):
        return [self.define("CUDOLFINX_SKIP_BUILD_TESTS", True)]
