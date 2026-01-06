# Copyright (C) 2024 Benjamin Pachev
#
# This file is part of cuDOLFINX
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Routines for manipulating generated FFCX code."""

import pathlib
from typing import Any

import numpy as np

from dolfinx import fem

__all__ = [
    "get_tabulate_tensor_sources",
    "get_wrapped_tabulate_tensors",
]

def get_tabulate_tensor_sources(form: fem.Form):
    """Given a compiled fem.Form, extract the C source code of the tabulate tensors
    """

    module_file = pathlib.Path(form.module.__file__)
    source_filename = module_file.name.split(".")[0] + ".c"
    source_file = module_file.parent.joinpath(source_filename)
    if not source_file.is_file():
        raise OSError("Could not find generated ffcx source file '{source_file}'!")

    tabulate_tensors = []
    parsing_tabulate = False
    parsing_header = False
    bracket_count = 0
    with open(source_file) as fp:
        for line in fp:
            if "tabulate_tensor_integral" in line and line.strip().startswith("void"):
                parsing_tabulate = True
                parsing_header = True
                tabulate_id = line.strip().split()[1].split("_")[-1].split("(")[0]
                tabulate_body = []
            elif parsing_header:
                if line.startswith("{"):
                    parsing_header = False
                    bracket_count += 1
            elif parsing_tabulate:
                if line.startswith("{"):
                    bracket_count += 1
                elif line.startswith("}"):
                    bracket_count -= 1
                if not bracket_count:
                    tabulate_tensors.append((tabulate_id, "".join(tabulate_body)))
                    parsing_tabulate = False
                else:
                    tabulate_body.append(line)
            elif "form_integrals_form" in line:
                if "{" in line:
                    arr = line.split("{")[-1].split("}")[0]
                    ordered_integral_ids = [
                        part.strip().split("_")[-1] for part in arr.split(",")
                    ]

    # map ids to order of appearance in tensor list
    id_order = {tabulate[0]: i for i, tabulate in enumerate(tabulate_tensors)}
    integral_tensor_indices = [id_order[integral_id] for integral_id in ordered_integral_ids]
    return tabulate_tensors, integral_tensor_indices

cuda_tabulate_tensor_header = """
    #define alignas(x)
    #define restrict __restrict__

    typedef unsigned char uint8_t;
    typedef unsigned int uint32_t;
    typedef double ufc_scalar_t;

    extern "C" __global__
    void tabulate_tensor_{factory_name}({scalar_type}* restrict A,
                                        const {scalar_type}* restrict w,
                                        const {scalar_type}* restrict c,
                                        const {geom_type}* restrict coordinate_dofs,
                                        const int* restrict entity_local_index,
                                        const uint8_t* restrict quadrature_permutation
                                        )
"""

def _convert_dtype_to_str(dtype: Any):
    """Convert numpy dtype to named C type
    """

    if dtype == np.float32:
        return "float"
    elif dtype == np.float64:
        return "double"
    else:
        raise TypeError(f"Unsupported dtype: '{dtype}'")

def get_wrapped_tabulate_tensors(form: fem.Form, backend="cuda"):
    """Given a fem.Form, wrap the tabulate tensors for use on a GPU
    """

    if backend != "cuda":
        raise NotImplementedError(f"Backend '{backend}' not yet supported.")

    # for now assume same type for form and mesh
    # this is typically the default
    geom_type = scalar_type = _convert_dtype_to_str(form.dtype)

    res = []
    sources, integral_tensor_indices = get_tabulate_tensor_sources(form)
    for id, body in sources:
        factory_name = "integral_" + id
        name = "tabulate_tensor_" + factory_name
        header = cuda_tabulate_tensor_header.format(
                scalar_type=scalar_type,
                geom_type=geom_type,
                factory_name=factory_name
                )
        wrapped_source = header + "{\n" + body + "}\n"
        res.append((name, wrapped_source))

    return res, integral_tensor_indices

