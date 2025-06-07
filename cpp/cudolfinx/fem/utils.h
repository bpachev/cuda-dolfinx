#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <cudolfinx/fem/CUDAForm.h>
#include <vector>

namespace dolfinx::fem {

// Create a restricted sparsity pattern
// This emulates the functionality in multiphenicsx
// However we don't want to depend on multiphenicsx just for this function
// TODO accelerate sparsity pattern computation with a CUDA kernel
template<typename T, std::floating_point U>
dolfinx::la::SparsityPattern compute_restricted_sparsity_pattern(std::shared_ptr<CUDAForm<T,U>> cuda_form)
{
  auto form = cuda_form->form();
  auto restriction = cuda_form->get_restriction();

  std::vector<std::int32_t> dofs0;
  std::vector<std::int32_t> dofs1;
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap0 = form->function_spaces()[0]->dofmap();
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap1 = form->function_spaces()[1]->dofmap();

  int num_cell_dofs = dofmap0->map().extent(1);
  std::shared_ptr mesh = form->mesh();
  const std::array index_maps{dofmap0->index_map,
                              dofmap1->index_map};
  const std::array bs
      = {dofmap0->index_map_bs(), dofmap1->index_map_bs()};

  // Create and build sparsity pattern
  la::SparsityPattern pattern(mesh->comm(), index_maps, bs);

  for (auto integral_type : form->integral_types()) {
    std::vector<int> ids = form->integral_ids(integral_type);
    switch (integral_type) {
      case dolfinx::fem::IntegralType::cell:
        dofs0.resize(num_cell_dofs);
        dofs1.resize(num_cell_dofs);
        for (auto id : ids) {
          auto cells = form->domain(integral_type, id);
          for (int cell : cells) {
            auto unrestricted_dofs0 = dofmap0->cell_dofs(cell);
            auto unrestricted_dofs1 = dofmap1->cell_dofs(cell);
            // this relies on runtime mapping, which means the integral
            // had better not go outside of the restriction
            // perhaps not the best plan. . . .
            for (int i = 0; i < num_cell_dofs; i++) {
              dofs0[i] = (*(restriction[0]))[unrestricted_dofs0[i]];
              dofs1[i] = (*(restriction[1]))[unrestricted_dofs1[i]]; 
            }
            // Although we are passing a span, the dofs get copied anyway 
            // so it is ok we are overwriting the local mapped dofs
            pattern.insert(std::span(dofs0.data(), dofs0.size()), std::span(dofs1.data(), dofs1.size()));
          }
        }
        break;
      case dolfinx::fem::IntegralType::exterior_facet:
        dofs0.resize(num_cell_dofs);
        dofs1.resize(num_cell_dofs);
        for (auto id : ids) {
          auto facets = form->domain(integral_type, id);
          for (int i = 0; i < facets.size(); i += 2) {
            int cell = facets[i];
            auto unrestricted_dofs0 = dofmap0->cell_dofs(cell);
            auto unrestricted_dofs1 = dofmap1->cell_dofs(cell);
            // this relies on runtime mapping, which means the integral
            // had better not go outside of the restriction
            // perhaps not the best plan. . . .
            for (int i = 0; i < num_cell_dofs; i++) {
              dofs0[i] = (*(restriction[0]))[unrestricted_dofs0[i]];
              dofs1[i] = (*(restriction[1]))[unrestricted_dofs1[i]]; 
            }
            // Although we are passing a span, the dofs get copied anyway 
            // so it is ok we are overwriting the local mapped dofs
            pattern.insert(std::span(dofs0.data(), dofs0.size()), std::span(dofs1.data(), dofs1.size()));
          }
        }
        break;
      case dolfinx::fem::IntegralType::interior_facet:
        dofs0.resize(2*num_cell_dofs);
        dofs1.resize(2*num_cell_dofs);
        for (auto id : ids) {
          auto facets = form->domain(integral_type, id);
          for (int i = 0; i < facets.size(); i += 4) {
            int cell0 = facets[i];
            int cell1 = facets[i+2];
            auto unrestricted_dofs00 = dofmap0->cell_dofs(cell0);
            auto unrestricted_dofs01 = dofmap0->cell_dofs(cell1);
            auto unrestricted_dofs10 = dofmap1->cell_dofs(cell0);
            auto unrestricted_dofs11 = dofmap1->cell_dofs(cell1);
            // this relies on runtime mapping, which means the integral
            // had better not go outside of the restriction
            // perhaps not the best plan. . . .
            for (int i = 0; i < num_cell_dofs; i++) {
              dofs0[i] = (*(restriction[0]))[unrestricted_dofs00[i]];
              dofs0[i+num_cell_dofs] = (*(restriction[0]))[unrestricted_dofs01[i]];
              dofs1[i] = (*(restriction[1]))[unrestricted_dofs10[i]];
              dofs1[i+num_cell_dofs] = (*(restriction[1]))[unrestricted_dofs11[i]];
            }
            // Although we are passing a span, the dofs get copied anyway 
            // so it is ok we are overwriting the local mapped dofs
            pattern.insert(std::span(dofs0.data(), dofs0.size()), std::span(dofs1.data(), dofs1.size()));
          }
        }
        break;
      default:
        break;
    }
  }
  return pattern;
}

} // end namespace dolfinx::fem
