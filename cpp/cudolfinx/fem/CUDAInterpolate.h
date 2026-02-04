#pragma once

#include <cstddef>
#include <cstdint>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/fem/interpolate.h>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace dolfinx::CUDA {

template<std::floating_point T>
void d_interpolate_same_map(T* u1,
                            T* u0,
                            int n0,
                            int n1, int C,
                            T* i_m, int* M0, int* M1);

/// Same-map interpolation on device.
///
/// @param[out] u1 Pointer to device-side coefficient array that will be updated
/// by the interpolation.
/// @param[in] u0 Pointer to device-side coefficient array to be interpolated FROM
/// @param[in] im_shape Shape of interpolation array
/// @param[in] num_cells No. of cells in the mesh
/// @param[in] i_m Pointer to device-side interpolation matrix with dim n1 x n0
/// @param[in] M1 Pointer to device-side DOF map for u1
/// @param[in] M0 Pointer to device-side DOF map for u0
template<std::floating_point T>
void interpolate_same_map(
                          CUdeviceptr ux1,
                          CUdeviceptr ux0,
                          std::array<std::size_t, 2> im_shape,
                          std::size_t num_cells,
                          CUdeviceptr i_m,
                          CUdeviceptr M1,
                          CUdeviceptr M0) {


  const std::size_t n1 = im_shape[0];
  const std::size_t n0 = im_shape[1];

  d_interpolate_same_map<T>((T*)ux1, (T*)ux0, n1, n0, num_cells, (T*)i_m, (int*)M1, (int*)M0);
}

/// @brief Create a global-to-cells DOF map for a given Function object.
/// 
/// @returns Map for Function u, of size dof_per_elem x num_cells. M[i,c] contains the index of the
/// global DOF that is mapped to the ith local DOF in cell `c`, in the reference ordering. 
/// @param[in] u Function object
template <dolfinx::scalar T, std::floating_point U>
std::vector<std::int32_t> create_interpolation_map(const dolfinx::fem::Function<T, U> &u) {

  auto V = u.function_space();
  auto mesh = V->mesh();
  auto element = V->element();

  const int tdim = mesh->topology()->dim();
  auto map = mesh->topology()->index_map(tdim);

  std::vector<std::int32_t> cells(map->size_local() + map->num_ghosts(), 0);
  std::iota(cells.begin(), cells.end(), 0);

  std::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable()->create_entity_permutations();
    cell_info = std::span(mesh->topology()->get_cell_permutation_info());
  }

  auto dofmap = V->dofmap();

  // Get block sizes and dof transformation operators
  const int bs = dofmap->bs();
  auto apply_dof_transformation = element->template dof_transformation_fn<std::int32_t>(
      dolfinx::fem::doftransform::transpose, false);

  std::size_t n = element->space_dimension();
  std::vector<std::int32_t> local_dofs(n);
  std::vector<std::int32_t> M(cells.size() * n);

  for (std::size_t c = 0; c < cells.size(); c++) {
    // Pack and transform cell dofs to reference ordering
    std::span<const std::int32_t> D = dofmap->cell_dofs(cells[c]);
    // local_dofs0 = [ 0, ..., k-1 ]
    std::iota(local_dofs.begin(), local_dofs.end(), 0);

    // Permute the vector [0, ..., k-1]
    apply_dof_transformation(local_dofs, cell_info, cells[c], 1);

    for (std::size_t i = 0; i < n; i++) {
      M[i * cells.size() + c] = D[local_dofs[i]];
    }
  }
  return M;
}
}
