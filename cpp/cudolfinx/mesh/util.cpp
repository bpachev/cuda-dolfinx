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

std::vector<std::int32_t> dolfinx::mesh::active_ghost_entities(
                std::span<const std::int32_t> active_local_entities,
                fem::IntegralType integral_type,
                std::shared_ptr<Topology> topology)
{
  std::vector<std::int32_t> ghost_entities;
  MPI_Comm comm = topology->comm();
  // no need for ghosting if there is only one process
  if (dolfinx::MPI::size(comm) == 1) return ghost_entities;
  int rank = dolfinx::MPI::rank(comm);
  int tdim = topology->dim();
  int ent_dim = (integral_type == fem::IntegralType::cell) ? tdim : tdim-1;
  // Step 1: determine the active entities which are ghosted on other processes 
  std::map<int, std::vector<std::int32_t>> dest_entities;
  auto imap = topology->index_map(ent_dim);
  int num_local_entities = imap->size_local();
  auto entity_ranks = imap->index_to_dest_ranks();
  int facet_increment = (integral_type == fem::IntegralType::interior_facet) ? 4 : 2;
  switch (integral_type) {
    case fem::IntegralType::cell:
      for (auto& cell : active_local_entities) {
        if (cell >= entity_ranks.num_nodes()) continue;
        for (auto& r : entity_ranks.links(cell)) {
          if (dest_entities.find(r) == dest_entities.end()) {
            dest_entities[r] = {cell};
          }
          else dest_entities[r].push_back(cell);
        }
      }
      break;
    case fem::IntegralType::interior_facet:
    case fem::IntegralType::exterior_facet: {
      auto c_to_f = topology->connectivity(tdim, tdim-1);
      if (!c_to_f) {
        topology->create_connectivity(tdim, tdim-1);
        c_to_f = topology->connectivity(tdim, tdim-1);
      }
      for (int i = 0; i < active_local_entities.size(); i += facet_increment) {
        auto cell = active_local_entities[i];
        auto facet_index = active_local_entities[i+1];
        auto facet = c_to_f->links(cell)[facet_index];
        if (facet >= entity_ranks.num_nodes()) continue;
        for (auto& r : entity_ranks.links(facet)) {
          if (dest_entities.find(r) == dest_entities.end()) {
            dest_entities[r] = {facet};
          }
          else dest_entities[r].push_back(facet);
        }
      }
    }
    default:
      break;
  }

  // Step 2: send those entities to the other processes
  std::vector<std::int64_t> indices_send_buffer;
  // construct list of destination MPI ranks
  std::vector<int> dest;
  std::vector<int> send_sizes;
  for (const auto& pair : dest_entities) {
    dest.push_back(pair.first);
    std::size_t num_inds = pair.second.size();
    send_sizes.push_back(num_inds);
    std::vector<std::int64_t> global_inds(num_inds);
    imap->local_to_global(pair.second, global_inds);
    for (const auto& i : global_inds)
      indices_send_buffer.push_back(i);
  }
  // get source ranks
  std::vector<int> src = dolfinx::MPI::compute_graph_edges_nbx(comm, dest);
  // Create neighbor communicator
  MPI_Comm neigh_comm;
  int ierr = MPI_Dist_graph_create_adjacent(
      comm, src.size(), src.data(), MPI_UNWEIGHTED, dest.size(),
      dest.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &neigh_comm);
  dolfinx::MPI::check_error(comm, ierr);
  // Share lengths of indices to be sent to each rank
  std::vector<int> recv_sizes(src.size(), 0);
  ierr = MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT,
                               recv_sizes.data(), 1, MPI_INT, neigh_comm);
  dolfinx::MPI::check_error(comm, ierr);
  // Prepare displacement arrays
  std::vector<int> send_disp(dest.size() + 1, 0);
  std::vector<int> recv_disp(src.size() + 1, 0);
  std::partial_sum(send_sizes.begin(), send_sizes.end(),
                   std::next(send_disp.begin()));
  std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                   std::next(recv_disp.begin()));
  // next steps - construct recv buffers and perform communication
  std::size_t recv_buf_size = recv_disp.back();
  // make sure that the buffer pointers actually get allocated
  std::vector<std::int64_t> indices_recv_buffer(recv_buf_size);
  ierr = MPI_Neighbor_alltoallv(indices_send_buffer.data(), send_sizes.data(),
                                send_disp.data(), MPI_INT64_T,
                                indices_recv_buffer.data(), recv_sizes.data(),
                                recv_disp.data(), MPI_INT64_T, neigh_comm);
  dolfinx::MPI::check_error(comm, ierr);
  // Step 3: Convert from global to local indices and do entity computation
  std::vector<std::int32_t> local_recv_indices(indices_recv_buffer.size());
  imap->global_to_local(indices_recv_buffer, local_recv_indices);

  switch (integral_type) {
    case fem::IntegralType::cell:
      return local_recv_indices;
      break;
    case fem::IntegralType::exterior_facet: {
      // Remove facets on internal inter-process boundary
      std::vector<std::int32_t> ext_facets;
      std::sort(local_recv_indices.begin(), local_recv_indices.end());
      std::ranges::set_difference(local_recv_indices, topology->interprocess_facets(),
                              std::back_inserter(ext_facets));
      auto c_to_f = topology->connectivity(tdim, tdim-1);
      auto f_to_c = topology->connectivity(tdim-1, tdim);
      ghost_entities.reserve(2*ext_facets.size());
      for (auto& facet : ext_facets) {
        if (f_to_c->num_links(facet) == 1) {
          auto pair =
            dolfinx::fem::impl::get_cell_facet_pairs<1>(facet, f_to_c->links(facet), *c_to_f);
          ghost_entities.insert(ghost_entities.end(), pair.begin(), pair.end());
        }
      }
      break;
    }
    case fem::IntegralType::interior_facet: {
      auto c_to_f = topology->connectivity(tdim, tdim-1);
      auto f_to_c = topology->connectivity(tdim-1, tdim);
      for (auto& facet : local_recv_indices) {
        if (f_to_c->num_links(facet) == 2) {
          auto pair =
            dolfinx::fem::impl::get_cell_facet_pairs<2>(facet, f_to_c->links(facet), *c_to_f);
          ghost_entities.insert(ghost_entities.end(), pair.begin(), pair.end());
        }
      }
    }
    default:
      break;
  }

  return ghost_entities;
}
