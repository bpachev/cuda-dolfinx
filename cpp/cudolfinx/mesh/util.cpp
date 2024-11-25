#include <cudolfinx/mesh/util.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Topology.h>

using namespace dolfinx;

std::vector<std::int32_t> dolfinx::mesh::ghost_exterior_facet_indices(std::shared_ptr<Topology> topology)
{
  const int tdim = topology->dim();
  auto f_to_c = topology->connectivity(tdim - 1, tdim);
  auto f_to_v = topology->connectivity(tdim-1, 0);
  if (!f_to_c)  {
    topology->create_connectivity(tdim-1, tdim);
    f_to_c = topology->connectivity(tdim-1, tdim);
  }
  if (!f_to_v) {
    topology->create_connectivity(tdim-1, 0);
    f_to_v = topology->connectivity(tdim-1, 0);
  }
  // Find all owned facets (not ghost) with only one attached cell
  auto facet_map = topology->index_map(tdim - 1);
  const int num_local_facets = facet_map->size_local();
  const int num_ghost_facets = facet_map->num_ghosts();
  const int num_local_vertices = topology->index_map(0)->size_local();
  std::vector<std::int32_t> facets;
  for (std::int32_t f = num_local_facets; f < num_local_facets+num_ghost_facets; ++f)
  {
    if (f_to_c->num_links(f) == 1) {
      // check to make sure at least one facet vertex is owned
      // otherwise this is not needed
      auto vertices = f_to_v->links(f);
      bool has_owned_vertex = false;
      for (int i = 0; i < vertices.size(); i++) {
        if (vertices[i] < num_local_vertices) has_owned_vertex = true;
      }
      if (has_owned_vertex) facets.push_back(f);
    }
  }
  // Remove facets on internal inter-process boundary
  std::vector<std::int32_t> ext_facets;
  std::ranges::set_difference(facets, topology->interprocess_facets(),
                              std::back_inserter(ext_facets));
  return ext_facets;
}

std::vector<std::int32_t> dolfinx::mesh::ghost_entities(
		dolfinx::fem::IntegralType integral_type, 
		std::shared_ptr<Topology> topology)
{
  std::vector<std::int32_t> ghost_entities; 
  int tdim = topology->dim();
  switch (integral_type) {
    case fem::IntegralType::cell:
      {
        auto cell_index_map = topology->index_map(tdim);
        int num_ghost_cells = cell_index_map->num_ghosts();
	int num_owned_cells = cell_index_map->size_local();
        ghost_entities.resize(num_ghost_cells);
        std::iota(ghost_entities.begin(), ghost_entities.end(), num_owned_cells);
      }
      break;
    case fem::IntegralType::exterior_facet:
      {
        auto ghost_exterior_facets = dolfinx::mesh::ghost_exterior_facet_indices(topology);
	ghost_entities.reserve(2*ghost_exterior_facets.size());
	auto c_to_f = topology->connectivity(tdim, tdim-1);
	auto f_to_c = topology->connectivity(tdim-1, tdim);
	for (std::int32_t f : ghost_exterior_facets) {
	  auto pair =
            dolfinx::fem::impl::get_cell_facet_pairs<1>(f, f_to_c->links(f), *c_to_f);
          ghost_entities.insert(ghost_entities.end(), pair.begin(), pair.end());
	}
      }
      break;
    case fem::IntegralType::interior_facet:
      {
        auto c_to_f = topology->connectivity(tdim, tdim-1);
        auto f_to_c = topology->connectivity(tdim-1, tdim);
        auto facet_map = topology->index_map(tdim-1);
        int num_local_facets = facet_map->size_local();
        int total_facets = num_local_facets + facet_map->num_ghosts();
        for (int f = num_local_facets; f < total_facets; f++) {
          if (f_to_c->num_links(f) == 2) {
            auto pairs =
                fem::impl::get_cell_facet_pairs<2>(f, f_to_c->links(f), *c_to_f);
            ghost_entities.insert(ghost_entities.end(), pairs.begin(), pairs.end());
	  }
	}
      }
    default:
      break;
  }
  return ghost_entities;
}
