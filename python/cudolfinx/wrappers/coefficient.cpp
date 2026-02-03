// Copyright (C) 2026 Chayanon Wichitrnithed
//
// This file is part of cuDOLFINX
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>
#include <cudolfinx/fem/CUDACoefficient.h>


namespace nb = nanobind;

template <typename T, typename U>
void declare_cuda_coefficient(nb::module_& m, std::string type)
{
  std::string pyclass_name = std::string("CUDACoefficient_") + type;
  nb::class_<dolfinx::fem::CUDACoefficient<T,U>>(m, pyclass_name.c_str(), "Device side function")
  .def(nb::init<std::shared_ptr<const dolfinx::fem::Function<T,U>>>(),
  "Create device side function from a given dolfinx Function object.")
  .def("interpolate",
       [](dolfinx::fem::CUDACoefficient<T,U>& self,
          dolfinx::fem::CUDACoefficient<T,U>& d_g) {
         self.interpolate(d_g);
       },
       "Interpolate from another Function with the same reference element mapping defined on the same mesh.")
  .def("values",
       [](dolfinx::fem::CUDACoefficient<T,U>& self) {
         auto v = self.values();
         return nb::ndarray<T, nb::numpy, nb::c_contig>(v.data(), {v.size()}).cast();
       },
       "Return a copy of the coefficient vector.");
}

namespace cudolfinx_wrappers
{
void coefficient(nb::module_& m) {
    declare_cuda_coefficient<double,double>(m, "float64");
    declare_cuda_coefficient<float,float>(m, "float32");
}
}
