# Copyright (C) 2024 Benjamin Pachev
#
# This file is part of cuDOLFINX
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Main module for CUDOLFINx."""

from importlib.metadata import version

from cudolfinx.assemble import CUDAAssembler
from cudolfinx.form import form
from cudolfinx.function import CUDAFunction
from cudolfinx.mesh import ghost_layer_mesh, ghost_layer_meshtags

__version__ = version("fenics-cudolfinx")

__all__ = [
    "CUDAAssembler",
    "CUDAFunction",
    "form",
    "ghost_layer_mesh",
    "ghost_layer_meshtags"
]
