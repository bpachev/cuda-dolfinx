// Copyright (C) 2024 Benjamin Pachev
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace cudolfinx_wrappers
{
void fem(nb::module_& m);
} // namespace cudolfinx_wrappers

NB_MODULE(cpp, m)
{
  // Create module for C++ wrappers
  m.doc() = "DOLFINx CUDA Python interface";
  m.attr("__version__") = CUDOLFINX_VERSION;

#ifdef NDEBUG
  nanobind::set_leak_warnings(false);
#endif
  // Create fem submodule [fem]
  nb::module_ fem = m.def_submodule("fem", "FEM module");
  cudolfinx_wrappers::fem(fem);
}