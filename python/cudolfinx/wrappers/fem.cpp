// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of cuDOLFINX
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <cstdint>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>
#include <cudolfinx/common/CUDA.h>
#include <cudolfinx/fem/CUDAAssembler.h>
#include <cudolfinx/fem/CUDADirichletBC.h>
#include <cudolfinx/fem/CUDADofMap.h>
#include <cudolfinx/fem/CUDAForm.h>
#include <cudolfinx/la/CUDAMatrix.h>
#include <cudolfinx/la/CUDAVector.h>
#include <cudolfinx/mesh/CUDAMesh.h>
#include <cudolfinx/mesh/util.h>
#include "caster_petsc.h"
#include <petsc4py/petsc4py.h>
#include <petscis.h>
#include <cuda.h>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <span>
#include <string>
#include <utility>
#include <ufcx.h>

namespace nb = nanobind;

namespace
{
template <typename T, typename = void>
struct geom_type
{
  typedef T value_type;
};
template <typename T>
struct geom_type<T, std::void_t<typename T::value_type>>
{
  typedef typename T::value_type value_type;
};

// declare meshtags function
template <typename T>
void declare_meshtags(nb::module_& m)
{
  m.def("ghost_layer_meshtags", dolfinx::mesh::ghost_layer_meshtags<T>,
        "Transfer meshtags to ghost layer mesh.");	  
}

// declare templated cuda-related objects
template <typename T>
void declare_cuda_templated_objects(nb::module_& m, std::string type)
{
  using U = typename dolfinx::scalar_value_type_t<T>;

  std::string pyclass_name = std::string("CUDAForm_") + type;
  nb::class_<dolfinx::fem::CUDAForm<T,U>>(m, pyclass_name.c_str(), "Form on GPU")
      .def(
          "__init__",
           [](dolfinx::fem::CUDAForm<T,U>* cf, const dolfinx::CUDA::Context& cuda_context,
              dolfinx::fem::Form<T,U>& form, std::uintptr_t ufcx_form,
	      std::vector<std::string>& tabulate_tensor_names, std::vector<std::string>& tabulate_tensor_sources,
	      std::vector<int>& integral_tensor_indices
	      )
             {
	       struct ufcx_form* p = reinterpret_cast<struct ufcx_form*>(ufcx_form);
               new (cf) dolfinx::fem::CUDAForm<T,U>(
                 cuda_context,
                 &form,
		 p,
		 tabulate_tensor_names,
		 tabulate_tensor_sources,
		 integral_tensor_indices
               );
             }, nb::arg("context"), nb::arg("form"), nb::arg("cuda_form"),
	     nb::arg("tabulate_tensor_names"), nb::arg("tabulate_tensor_sources"),
	     nb::arg("integral_tensor_indices")
	     )
      .def(
          "compile",
          [](dolfinx::fem::CUDAForm<T,U>& cf, const dolfinx::CUDA::Context& cuda_context,
             int32_t max_threads_per_block, int32_t min_blocks_per_multiprocessor)
             {
               cf.compile(cuda_context, max_threads_per_block,
                          min_blocks_per_multiprocessor, dolfinx::fem::assembly_kernel_type::ASSEMBLY_KERNEL_GLOBAL);
             }, nb::arg("context"), nb::arg("max_threads_per_block"), nb::arg("min_blocks_per_multiprocessor"))
      .def(
          "set_restriction",
	  [](dolfinx::fem::CUDAForm<T,U>& cf, std::vector<int32_t> offsets,
       std::vector<int32_t> ghost_offsets,
	     std::vector<std::vector<int32_t>>& restricted_inds)
	     {
	       if (restricted_inds.size() != offsets.size()) {
                 throw std::runtime_error("Length of restricted inds and offset lists must match!");
               }
               std::vector<std::shared_ptr<std::map<::int32_t, std::int32_t>>> restrictions;
               for (int i = 0; i < restricted_inds.size(); i++) {
                 auto m = std::make_shared<std::map<std::int32_t, std::int32_t>>();
                 for (int j = 0; j < restricted_inds[i].size(); j++) (*m)[restricted_inds[i][j]] = j;
                 restrictions.push_back(m);
               }
               cf.set_restriction(offsets, ghost_offsets, restrictions);
	     }, nb::arg("offsets"), nb::arg("ghost_offsets"), nb::arg("restricted_inds"))
      .def_prop_ro("compiled", &dolfinx::fem::CUDAForm<T,U>::compiled)
      .def_prop_ro("restricted", &dolfinx::fem::CUDAForm<T,U>::restricted)
      .def("to_device", &dolfinx::fem::CUDAForm<T,U>::to_device);

  pyclass_name = std::string("CUDADirichletBC_") + type;
  nb::class_<dolfinx::fem::CUDADirichletBC<T,U>>(m, pyclass_name.c_str(),
                                                 "Dirichlet BC on GPU")
      .def(
          "__init__",
          [](dolfinx::fem::CUDADirichletBC<T,U>* bc, const dolfinx::CUDA::Context& cuda_context,
             const dolfinx::fem::FunctionSpace<T>& V,
             const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T,U>>>& bcs)
             {
               new (bc) dolfinx::fem::CUDADirichletBC<T,U>(
                   cuda_context, V, bcs);
             },
          nb::arg("context"), nb::arg("V"), nb::arg("bcs"))
      .def("update", &dolfinx::fem::CUDADirichletBC<T,U>::update, nb::arg("bcs"));

  std::string pyclass_cumesh_name = std::string("CUDAMesh_") + type;
  nb::class_<dolfinx::mesh::CUDAMesh<T>>(m, pyclass_cumesh_name.c_str(),
                                         "Mesh object on GPU")
      .def(
          "__init__",
          [](dolfinx::mesh::CUDAMesh<T>* cumesh,
             const dolfinx::mesh::Mesh<T>& mesh) {
            new (cumesh) dolfinx::mesh::CUDAMesh<T>(mesh);
          },
          nb::arg("mesh"));

  m.def("ghost_layer_mesh", dolfinx::mesh::ghost_layer_mesh<T>,
	"Create mesh with extra layer of ghost cells.");

}

// Declare the nontemplated CUDA wrappers
void declare_cuda_objects(nb::module_& m)
{
  import_petsc4py();

  nb::class_<dolfinx::CUDA::Context>(m, "CUDAContext", "CUDA Context")
      .def("__init__", [](dolfinx::CUDA::Context* c) { new (c) dolfinx::CUDA::Context();});

  nb::class_<dolfinx::la::CUDAMatrix>(m, "CUDAMatrix", "Matrix object on GPU")
      .def(
          "__init__",
          [](dolfinx::la::CUDAMatrix* cumat, const dolfinx::CUDA::Context& cuda_context, Mat A) {
            new (cumat) dolfinx::la::CUDAMatrix(cuda_context, A, false, true);
          }, nb::arg("context"), nb::arg("A"))
      .def("debug_dump", &dolfinx::la::CUDAMatrix::debug_dump)
      .def("to_host",
          [](dolfinx::la::CUDAMatrix& cumat, const dolfinx::CUDA::Context& cuda_context)
          {
            cumat.copy_matrix_values_to_host(cuda_context);
            cumat.apply(MAT_FINAL_ASSEMBLY);
          }, nb::arg("cuda_context"), "Copy matrix values to host.")
      .def_prop_ro("mat",
          [](dolfinx::la::CUDAMatrix& cumat) {
            Mat A = cumat.mat();
            PyObject* obj = PyPetscMat_New(A);
            PetscObjectDereference((PetscObject)A);
            return nb::borrow(obj);
          });

  nb::class_<dolfinx::la::CUDAVector>(m, "CUDAVector", "Vector object on GPU")
      .def(
          "__init__",
          [](dolfinx::la::CUDAVector* cuvec, const dolfinx::CUDA::Context& cuda_context, Vec x) {
            new (cuvec) dolfinx::la::CUDAVector(cuda_context, x, false, false);
          }, nb::arg("context"), nb::arg("x"))
      .def("to_host", &dolfinx::la::CUDAVector::copy_vector_values_to_host)
      .def_prop_ro("vector",
          [](dolfinx::la::CUDAVector& cuvec) {
            Vec b = cuvec.vector();
            PyObject* obj = PyPetscVec_New(b);
            PetscObjectDereference((PetscObject)b);
            return nb::borrow(obj);
          });

  auto assembler_class = nb::class_<dolfinx::fem::CUDAAssembler>(m, "CUDAAssembler", "Assembler object")
      .def(
          "__init__",
          [](dolfinx::fem::CUDAAssembler* assembler, const dolfinx::CUDA::Context& cuda_context,
             const char* cudasrcdir) {
            bool debug = true, verbose = false;
            CUjit_target target = dolfinx::CUDA::get_cujit_target(cuda_context);
            new (assembler) dolfinx::fem::CUDAAssembler(cuda_context, target, debug, cudasrcdir, verbose);
          }, nb::arg("context"), nb::arg("cudasrcdir"));
  
}

// Declare some functions that 
// simplify the process of CUDA assembly in Python
template <typename T, typename U>
void declare_cuda_funcs(nb::module_& m)
{

  m.def("pack_coefficients",
        [](const dolfinx::CUDA::Context& cuda_context, dolfinx::fem::CUDAAssembler& assembler,
          dolfinx::fem::CUDAForm<T,U>& cuda_form)
        {
          assembler.pack_coefficients(cuda_context, cuda_form.coefficients());
        },
        nb::arg("context"), nb::arg("assembler"), nb::arg("cuda_form"), "Pack form coefficients on device.");

  m.def("pack_coefficients",
       [](const dolfinx::CUDA::Context& cuda_context, dolfinx::fem::CUDAAssembler& assembler,
          dolfinx::fem::CUDAForm<T,U>& cuda_form, std::vector<std::shared_ptr<dolfinx::fem::Function<T,U>>>& coefficients)
       {
         if (!coefficients.size()) {
           // nothing to do
           return;
         }

         assembler.pack_coefficients(cuda_context, cuda_form.coefficients(), coefficients);
       },
       nb::arg("context"), nb::arg("assembler"), nb::arg("cuda_form"), nb::arg("coefficients"),
       "Pack a given subset of form coefficients on device");

  m.def("zero_matrix_entries",
        [](const dolfinx::CUDA::Context& cuda_context, dolfinx::fem::CUDAAssembler& assembler,
           dolfinx::la::CUDAMatrix& cuda_A) {
          
          assembler.zero_matrix_entries(cuda_context, cuda_A);
        }, nb::arg("context"), nb::arg("assembler"), nb::arg("A"),
        "Zero matrix entries"
  );

  m.def("assemble_matrix_on_device",
        [](const dolfinx::CUDA::Context& cuda_context, dolfinx::fem::CUDAAssembler& assembler,
           dolfinx::fem::CUDAForm<T,U>& cuda_form, dolfinx::mesh::CUDAMesh<U>& cuda_mesh,
           dolfinx::la::CUDAMatrix& cuda_A, dolfinx::fem::CUDADirichletBC<T,U>& cuda_bc0,
           dolfinx::fem::CUDADirichletBC<T,U>& cuda_bc1) {
          // Extract constant and coefficient data
          std::shared_ptr<const dolfinx::fem::CUDADofMap> cuda_dofmap0 =
            cuda_form.dofmap(0);
          std::shared_ptr<const dolfinx::fem::CUDADofMap> cuda_dofmap1 =
            cuda_form.dofmap(1);

          // not needed for global assembly kernel
          /*assembler.compute_lookup_tables(
            cuda_context, *cuda_dofmap0, *cuda_dofmap1,
            cuda_bc0, cuda_bc1, cuda_a_form_integrals, cuda_A, false);*/
          assembler.assemble_matrix(
            cuda_context, cuda_mesh, *cuda_dofmap0, *cuda_dofmap1,
            cuda_bc0, cuda_bc1, cuda_form.integrals(),
            cuda_form.constants(), cuda_form.coefficients(),
            cuda_A, false);
          assembler.set_diagonal(cuda_context, cuda_A, cuda_bc0);
        },
        nb::arg("context"), nb::arg("assembler"), nb::arg("form"), nb::arg("mesh"),
        nb::arg("A"), nb::arg("bcs0"), nb::arg("bcs1"), "Assemble matrix on GPU."
  );

  // TODO: replace this function with PETSc API call
  m.def("zero_vector_entries",
	[](const dolfinx::CUDA::Context& cuda_context, dolfinx::fem::CUDAAssembler& assembler,
	   dolfinx::la::CUDAVector& cuda_b)
	{
          assembler.zero_vector_entries(cuda_context, cuda_b);
	},
	nb::arg("context"), nb::arg("assembler"), nb::arg("b"),
	"Zero vector entries"
  );

  m.def("assemble_vector_on_device",
        [](const dolfinx::CUDA::Context& cuda_context, dolfinx::fem::CUDAAssembler& assembler,
           dolfinx::fem::CUDAForm<T,U>& cuda_form,
           dolfinx::mesh::CUDAMesh<U>& cuda_mesh,
           dolfinx::la::CUDAVector& cuda_b)
         {
          
          std::shared_ptr<const dolfinx::fem::CUDADofMap> cuda_dofmap0 =
            cuda_form.dofmap(0);
          
          assembler.assemble_vector(
             cuda_context, cuda_mesh, *cuda_dofmap0,
             cuda_form.integrals(), cuda_form.constants(),
             cuda_form.coefficients(), cuda_b, false);
          
        },
        nb::arg("context"), nb::arg("assembler"), nb::arg("form"), nb::arg("mesh"), nb::arg("b"),
        "Assemble vector on GPU."
  );

  m.def("apply_lifting_on_device",
        [](const dolfinx::CUDA::Context& cuda_context, dolfinx::fem::CUDAAssembler& assembler,
           std::vector<std::shared_ptr<dolfinx::fem::CUDAForm<T,U>>>& cuda_form,
           dolfinx::mesh::CUDAMesh<U>& cuda_mesh,
           dolfinx::la::CUDAVector& cuda_b,
           std::vector<std::shared_ptr<const dolfinx::fem::CUDADirichletBC<T,U>>>& bcs,
           std::vector<std::shared_ptr<dolfinx::la::CUDAVector>>& cuda_x0,
           float scale)
        {
          bool missing_x0 = (cuda_x0.size() == 0);
          if (bcs.size() != cuda_form.size()) throw std::runtime_error("Number of bc lists must match number of forms!");
          if (!missing_x0 && (cuda_x0.size() != cuda_form.size())) 
            throw std::runtime_error("Number of x0 vectors must match number of forms!");

          for (size_t i = 0; i < cuda_form.size(); i++) {
            auto form = cuda_form[i];
            std::shared_ptr<dolfinx::la::CUDAVector> x0 = (missing_x0) ? nullptr : cuda_x0[i]; 
            assembler.lift_bc(
              cuda_context, cuda_mesh, *form->dofmap(0), *form->dofmap(1),
              form->integrals(), form->constants(), form->coefficients(),
              *bcs[i], x0, scale, cuda_b, false
            );
          }

        },
        nb::arg("context"), nb::arg("assembler"), nb::arg("form"), nb::arg("mesh"), nb::arg("b"),
        nb::arg("bcs"), nb::arg("x0"), nb::arg("scale"),
        "Apply lifting on GPU" 
  );

  m.def("set_bc_on_device",
        [](const dolfinx::CUDA::Context& cuda_context, dolfinx::fem::CUDAAssembler& assembler,
           dolfinx::la::CUDAVector& cuda_b,
           std::shared_ptr<const dolfinx::fem::CUDADirichletBC<T,U>> bc0,
           std::shared_ptr<dolfinx::la::CUDAVector> cuda_x0,
           float scale)
        {
          assembler.set_bc(cuda_context, *bc0, cuda_x0, scale, cuda_b);
        },
        nb::arg("context"), nb::arg("assembler"), nb::arg("b"),
        nb::arg("bcs"), nb::arg("x0"), nb::arg("scale"),
        "Set boundary conditions on GPU"
  );

  m.def("set_bc_on_device",
        [](const dolfinx::CUDA::Context& cuda_context, dolfinx::fem::CUDAAssembler& assembler,
           dolfinx::la::CUDAVector& cuda_b,
           std::shared_ptr<const dolfinx::fem::CUDADirichletBC<T,U>> bc0,
           float scale)
        {
          std::shared_ptr<dolfinx::la::CUDAVector> x0 = nullptr;
          assembler.set_bc(cuda_context, *bc0, x0, scale, cuda_b);
        },
        nb::arg("context"), nb::arg("assembler"), nb::arg("b"),
        nb::arg("bcs"), nb::arg("scale"),
        "Set boundary conditions on GPU"
  );
 
}


} // namespace

namespace cudolfinx_wrappers
{

void fem(nb::module_& m)
{

  declare_cuda_templated_objects<float>(m, "float32");
  declare_cuda_templated_objects<double>(m, "float64");
  declare_cuda_objects(m);
  // TODO dynamically determine PETSc float type to assess which of these to invoke
  // defaults to assuming PETSc is built with double precision
  //declare_cuda_funcs<float, float>(m);
  declare_cuda_funcs<double, double>(m);
  declare_meshtags<std::int8_t>(m);
  declare_meshtags<std::int32_t>(m);
  declare_meshtags<std::int64_t>(m);
  declare_meshtags<double>(m);
}
} // namespace cudolfinx_wrappers
