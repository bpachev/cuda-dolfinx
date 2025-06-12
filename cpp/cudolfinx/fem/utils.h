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

  std::shared_ptr<const dolfinx::fem::DofMap> dofmap0 = form->function_spaces()[0]->dofmap();
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap1 = form->function_spaces()[1]->dofmap();

  int num_cell_dofs = dofmap0->map().extent(1);
  std::shared_ptr mesh = form->mesh();
  const std::array index_maps{dofmap0->index_map,
                              dofmap1->index_map};
  const std::array bs
      = {dofmap0->index_map_bs(), dofmap1->index_map_bs()};
  const std::array dofmaps = {dofmap0, dofmap1};
  std::array<std::vector<std::int32_t>, 2> restricted_cell_dofs;
  std::array<std::vector<std::int32_t>, 2> restricted_cell_bounds;
  std::array<std::vector<std::int32_t>, 2> insertion_dofs;

  for (std::size_t d = 0; d < 2; d++) {
    auto cell_map = dofmaps[d]->map();
    int num_cells = cell_map.extent(0);
    const auto& restriction_map = *restriction[d];
    restricted_cell_bounds[d].reserve(num_cells+1);
    restricted_cell_bounds[d].push_back(0);
    for (int cell = 0; cell < num_cells; cell++) {
      for (auto dof: dofmaps[d]->cell_dofs(cell)) {
        if (restriction_map.find(dof) != restriction_map.end()) {
          restricted_cell_dofs[d].push_back(restriction_map.at(dof));
        }
        restricted_cell_bounds[d].push_back(restricted_cell_dofs.size());
      }
    }
  } 

  // Create and build sparsity pattern
  la::SparsityPattern pattern(mesh->comm(), index_maps, bs);

  for (auto integral_type : form->integral_types()) {
    std::vector<int> ids = form->integral_ids(integral_type);
    if (integral_type == dolfinx::fem::IntegralType::interior_facet) {
      for (auto id: ids) {
        auto entities = form->domain(integral_type, id);
        for (std::size_t i = 0; i < entities.size(); i+=4) {
          int cell0 = entities[i];
          int cell1 = entities[i+2];
          for (std::size_t d = 0; d < 2; d++) {
            auto cell_dofs0 = std::span(restricted_cell_dofs[d].data() + restricted_cell_bounds[d][cell0],
                                      restricted_cell_bounds[d][cell0+1] - restricted_cell_bounds[d][cell0]);
            auto cell_dofs1 = std::span(restricted_cell_dofs[d].data() + restricted_cell_bounds[d][cell1],
                                      restricted_cell_bounds[d][cell1+1] - restricted_cell_bounds[d][cell1]);
            insertion_dofs[d].resize(cell_dofs0.size() + cell_dofs1.size());
            std::copy(cell_dofs0.begin(), cell_dofs0.end(), insertion_dofs[d].begin());
            std::copy(cell_dofs1.begin(), cell_dofs1.end(),
                      std::next(insertion_dofs[d].begin(), cell_dofs0.size()));
          }
          pattern.insert(insertion_dofs[0], insertion_dofs[1]);
        }
      }
    }
    else {
      int increment = (integral_type == dolfinx::fem::IntegralType::exterior_facet) ? 2 : 1;
      for (auto id: ids) {
        auto entities = form->domain(integral_type, id);
        for (std::size_t i = 0; i < entities.size(); i+=increment) {
          int cell = entities[i];
          auto cell_dofs0 = std::span(restricted_cell_dofs[0].data() + restricted_cell_bounds[0][cell],
                                      restricted_cell_bounds[0][cell+1] - restricted_cell_bounds[0][cell]);
          auto cell_dofs1 = std::span(restricted_cell_dofs[1].data() + restricted_cell_bounds[1][cell],
                                      restricted_cell_bounds[1][cell+1] - restricted_cell_bounds[1][cell]);
          pattern.insert(cell_dofs0, cell_dofs1);
        }
      }
    }
  }
  return pattern;
}

} // end namespace dolfinx::fem
