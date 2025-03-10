// Copyright (C) 2024 Benjamin Pachev, Igor Baratta
//
// This file is part of cuDOLFINX
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
#pragma once

#include <basix/finite-element.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/Form.h>
#include <algorithm>

namespace dolfinx {
namespace mesh {

/// Given an input mesh, create a version of the mesh with 
/// an additional layer of ghost cells to faciliate assembly
/// across multiple GPUs.
template <std::floating_point T>
dolfinx::mesh::Mesh<T> ghost_layer_mesh(dolfinx::mesh::Mesh<T>& mesh,
                                        dolfinx::fem::CoordinateElement<T> coord_element)
{
  int tdim = mesh.topology()->dim();
  int gdim = mesh.geometry().dim();
  std::size_t ncells = mesh.topology()->index_map(tdim)->size_local();
  std::size_t num_vertices = mesh.topology()->index_map(0)->size_local();

  // Find which local vertices are ghosted elsewhere
  auto vertex_destinations = mesh.topology()->index_map(0)->index_to_dest_ranks();

  // Map from any local cells to processes where they should be ghosted
  std::map<int, std::vector<int>> cell_to_dests;
  auto c_to_v = mesh.topology()->connectivity(tdim, 0);

  std::vector<int> cdests;
  for (std::size_t c = 0; c < ncells; ++c)
  {
    cdests.clear();
    for (auto v : c_to_v->links(c))
    {
      auto vdest = vertex_destinations.links(v);
      for (int dest : vdest)
        cdests.push_back(dest);
    }
    std::sort(cdests.begin(), cdests.end());
    cdests.erase(std::unique(cdests.begin(), cdests.end()), cdests.end());
    if (!cdests.empty())
      cell_to_dests[c] = cdests;
  }

  spdlog::info("cell_to_dests= {}, ncells = {}", cell_to_dests.size(), ncells);
  auto partitioner
      = [cell_to_dests, ncells](MPI_Comm comm, int nparts,
                                const std::vector<dolfinx::mesh::CellType>& cell_types,
                                const std::vector<std::span<const std::int64_t>>& cells)
  {
    int rank = dolfinx::MPI::rank(comm);
    std::vector<std::int32_t> dests;
    std::vector<int> offsets = {0};
    for (int c = 0; c < ncells; ++c)
    {
      dests.push_back(rank);
      if (auto it = cell_to_dests.find(c); it != cell_to_dests.end())
        dests.insert(dests.end(), it->second.begin(), it->second.end());

      // Ghost to other processes
      offsets.push_back(dests.size());
    }
    return dolfinx::graph::AdjacencyList<std::int32_t>(std::move(dests), std::move(offsets));
  };

  // The number of coordinates per vertex is ALWAYS 3, even for two-dimensional meshes
  std::array<std::size_t, 2> xshape = {num_vertices, 3};
  std::span<const T> x(mesh.geometry().x().data(), xshape[0] * xshape[1]);

  auto dofmap = mesh.geometry().dofmap();
  auto imap = mesh.geometry().index_map();
  // TODO figure how to properly support both tensor product elements
  // and regular elements
  /*std::vector<std::int32_t> permuted_dofmap;
  std::vector<int> perm = basix::tp_dof_ordering(
      basix::element::family::P, dolfinx::mesh::cell_type_to_basix_type(coord_element.cell_shape()),
      coord_element.degree(), coord_element.variant(), basix::element::dpc_variant::unset, false);
  for (std::size_t c = 0; c < dofmap.extent(0); ++c)
  {
    auto cell_dofs = std::submdspan(dofmap, c, std::full_extent);
    for (int i = 0; i < dofmap.extent(1); ++i)
      permuted_dofmap.push_back(cell_dofs(perm[i]));
  }
  std::vector<std::int64_t> permuted_dofmap_global(permuted_dofmap.size());
  imap->local_to_global(permuted_dofmap, permuted_dofmap_global);

  auto new_mesh
      = dolfinx::mesh::create_mesh(mesh.comm(), mesh.comm(), std::span(permuted_dofmap_global),
                                   coord_element, mesh.comm(), x, xshape, partitioner);*/
  std::vector<std::int32_t> input_dofmap;
  for (std::size_t c = 0; c < dofmap.extent(0); ++c) {
    auto cell_dofs = std::submdspan(dofmap, c, std::full_extent);
    for (int i = 0; i < dofmap.extent(1); ++i)
      input_dofmap.push_back(cell_dofs(i));
  }
  std::vector<std::int64_t> input_dofmap_global(input_dofmap.size());
  imap->local_to_global(input_dofmap, input_dofmap_global);
  auto new_mesh
      = create_mesh(mesh.comm(), mesh.comm(), std::span(input_dofmap_global), coord_element,
		                   mesh.comm(), x, xshape, partitioner);
  return new_mesh;
}

/// Return indices of ghost (non-owned) exterior facets
std::vector<std::int32_t> ghost_exterior_facet_indices(std::shared_ptr<Topology> topology);

/// Compute ghost entities given the integral type
std::vector<std::int32_t> ghost_entities(
                fem::IntegralType integral_type,
                std::shared_ptr<Topology> topology);

/// Given a MeshTags object for the original mesh without an extra ghost layer,
/// create a corresponding MeshTags object that will function correctly with 
/// the ghost layer mesh. This relies on original_cell_index in the Topology class,
/// so the MeshTags object MUST correspond to the mesh passed to ghost_layer_mesh!
template <typename T>
dolfinx::mesh::MeshTags<T> ghost_layer_meshtags(dolfinx::mesh::MeshTags<T>& meshtags,
		std::shared_ptr<Topology> ghosted_mesh_topology)
{
  std::shared_ptr<const Topology> topology = meshtags.topology();
  int tdim = topology->dim();
  // Get dimension of tagged entities
  int ent_dim = meshtags.dim();
  // Get original cell indices
  // TODO when mixed topology is implemented in DOLFINx, this will need to be updated
  std::vector<std::int64_t> original_cell_index = ghosted_mesh_topology->original_cell_index[0];
  int num_local_cells = topology->index_map(tdim)->size_local();
  if (num_local_cells != ghosted_mesh_topology->index_map(tdim)->size_local())
    throw std::runtime_error("Size mismatch in number of local cells between original and ghosted meshes!");
  
  auto cell_local_range = topology->index_map(tdim)->local_range();
  std::vector<std::int32_t> cell_map(num_local_cells);
  
  for (int i = 0; i < num_local_cells; i++) {
    int orig_cell = original_cell_index[i] - cell_local_range[0];
    if (orig_cell < 0 || orig_cell >= cell_map.size())
      throw std::runtime_error("Index out of bounds when constructing cell map!");
    cell_map[orig_cell] = i;
  }

  auto tagged_entities = meshtags.indices();
  auto tags = meshtags.values();
  std::vector<std::int32_t> input_entities;
  std::vector<T> input_values;
  // If the MeshTags object is for cells, it's easy
  if (ent_dim == tdim) {
    input_entities.reserve(tagged_entities.size());
    input_values.reserve(tags.size());
    for (int i = 0; i < tagged_entities.size(); i++) {
      input_entities.push_back(cell_map[tagged_entities[i]]);
      input_values.push_back(tags[i]);
    }
  } 
  else {
    // we have to do some monkey business here
    // first off, we loop over cells and match vertices
    auto cell_verts = topology->connectivity(tdim, 0);
    auto ghosted_cell_verts = ghosted_mesh_topology->connectivity(tdim, 0);

    std::map<std::int32_t, std::int32_t> vert_map;

    for (int cell = 0; cell < num_local_cells; cell++) {
      auto verts1 = cell_verts->links(cell);
      auto verts2 = ghosted_cell_verts->links(cell_map[cell]);
      // This is the crucial bit
      // We assume that the ordering of cell vertices is preserved
      // Without this we cannot match vertices unless we use the coordinates
      for (int j = 0; j < verts1.size(); j++) {
        vert_map[verts1[j]] = verts2[j];
      }
    }

    if (ent_dim == 0) {
      // We have the map constructed
      for (int i = 0; i < tagged_entities.size(); i++) {
        input_entities.push_back(vert_map[tagged_entities[i]]);
	input_values.push_back(tags[i]);
      }
    }
    else {
      auto ent_verts = topology->connectivity(ent_dim, 0);
      std::vector<std::int32_t> mapped_entities;
      for (const auto& ent : tagged_entities) {
	for (const auto& vert : ent_verts->links(ent)) {
	  if (vert_map.find(vert) == vert_map.end())
	    throw std::runtime_error("Unmapped vertex in tagged entitity!");
          mapped_entities.push_back(vert_map[vert]);
	}
      }

      ghosted_mesh_topology->create_connectivity(ent_dim, 0);
      std::vector<std::int32_t> entity_indices = dolfinx::mesh::entities_to_index(
		      *ghosted_mesh_topology, ent_dim, mapped_entities);

      for (int i = 0; i < entity_indices.size(); i++) {
        input_entities.push_back(entity_indices[i]);
        input_values.push_back(tags[i]);
      }
    }
  }

  int rank = dolfinx::MPI::rank(ghosted_mesh_topology->comm());
  std::map<int, std::vector<std::int32_t>> dest_entities;
  auto entity_ranks = ghosted_mesh_topology->index_map(ent_dim)->index_to_dest_ranks();
  for (int i = 0; i < input_entities.size(); i++) {
    std::int32_t ent = input_entities[i];
    if (ent >= entity_ranks.num_nodes()) continue;
    for (auto& r : entity_ranks.links(ent)) {
      if (dest_entities.find(r) == dest_entities.end()) {
        dest_entities[r] = {i};
      }
      else dest_entites[r].push_back(i);
    }
  }

  // get source ranks
  //std::vector<int> src = dolfinx::MPI::compute_graph_edges_nbx(comm, dest);
  //MPI_Comm neigh_comm;


  // Ensure entities/values are sorted before creating new MeshTags object
  std::vector<std::pair<std::int32_t, T>> combined;
  for (int i = 0; i < input_entities.size(); i++)
    combined.emplace_back(input_entities[i], input_values[i]);

  std::sort(combined.begin(), combined.end(), 
             [](const std::pair<std::int32_t, T>& a, const std::pair<std::int32_t, T>& b) {
               return a.first < b.first;
	     });

  input_entities.clear();
  input_values.clear();
  for (const auto& pair : combined) {
    input_entities.push_back(pair.first);
    input_values.push_back(pair.second);
  }

  return dolfinx::mesh::MeshTags<T>(ghosted_mesh_topology, ent_dim,
             std::move(input_entities), std::move(input_values));
}

} // namespace mesh
} // namespace dolfinx
