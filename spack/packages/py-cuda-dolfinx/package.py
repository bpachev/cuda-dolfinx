# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.package import *


class PyCudaDolfinx(PythonPackage):
    """Python interface for CUDA acceleration of DOLFINx in the FEniCS project."""

    homepage = "https://github.com/bpachev/cuda-dolfinx"
    url = "https://github.com/bpachev/cuda-dolfinx/archive/refs/tags/v0.9.0.zip"
    git = "https://github.com/bpachev/cuda-dolfinx.git"

    maintainers("bpachev")

    license("LGPL-3.0-only")
    version("main", branch="main")
    version("0.9.0", sha256="5c93155e58eee139985e9e9341cf7d8b29f8c9cbc51ccdf05134cdfb70ae105d")

    depends_on("cxx", type="build")
    depends_on("cmake@3.21:", when="@0.9:", type="build")
    depends_on("cuda-dolfinx@main", when="@main")
    depends_on("cuda-dolfinx@0.9.0", when="@0.9.0")
    depends_on("pkgconfig", type="build")
    depends_on("py-nanobind@2:", when="@0.9:", type="build")
    depends_on("py-scikit-build-core+pyproject@0.5:", when="@0.9:", type="build")

    build_directory = "python"

