// Copyright (C) 2024 Benjamin Pachev, Igor Baratta
//
// This file is part of cuDOLFINX
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
#pragma once

#include <basix/finite-element.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/Form.h>

namespace dolfinx {
namespace mesh {

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
  std::cout << "cell_to_dests " << cell_to_dests.size() << " ncells " << ncells << std::endl;
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
    std::cout << "Calling partitioner dests size " << dests.size() << std::endl;
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

// Return indices of ghost (non-owned) exterior facets
std::vector<std::int32_t> ghost_exterior_facet_indices(std::shared_ptr<Topology> topology);

// Compute ghost entities given the integral type
std::vector<std::int32_t> ghost_entities(
                fem::IntegralType integral_type,
                std::shared_ptr<Topology> topology);

} // namespace mesh
} // namespace dolfinx
