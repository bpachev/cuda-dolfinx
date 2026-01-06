# Copyright (C) 2024 Benjamin Pachev
#
# This file is part of cuDOLFINX
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Main module for cuDOLFINx"""

from importlib.metadata import version

from cudolfinx.assemble import CUDAAssembler
from cudolfinx.form import form
from cudolfinx.mesh import ghost_layer_mesh, ghost_layer_meshtags

__version__ = version("fenics-cudolfinx")

__all__ = [
    "CUDAAssembler",
    "form",
    "ghost_layer_mesh",
    "ghost_layer_meshtags"
]
