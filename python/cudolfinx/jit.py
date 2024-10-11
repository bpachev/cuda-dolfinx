"""Routines for manipulating generated FFCX code
"""

from dolfinx import fem
import pathlib

def get_tabulate_tensor_sources(form: fem.Form):
    """Given a compiled fem.Form, extract the C source code of the tabulate tensors
    """

    module_file = pathlib.Path(form.module.__file__)
    source_filename = module_file.name.split(".")[0] + ".c"
    source_file = module_file.parent.joinpath(source_filename)
    if not source_file.is_file():
        raise IOError("Could not find generated ffcx source file '{source_file}'!")

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
                if line.startswith("{"): bracket_count += 1
                elif line.startswith("}"): bracket_count -= 1
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

    id_order = {integral_id: i for i, integral_id in enumerate(ordered_integral_ids)}
    return sorted(tabulate_tensors, key=lambda tabulate: id_order[tabulate[0]])

