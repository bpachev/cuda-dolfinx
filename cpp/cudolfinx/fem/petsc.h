// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of cuDOLFINX
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <concepts>
#include <cudolfinx/fem/CUDAForm.h>
#include <cudolfinx/fem/utils.h>
#include <cudolfinx/la/petsc.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/la/petsc.h>
#include <map>
#include <memory>
#include <petscmat.h>
#include <petscvec.h>
#include <span>
#include <utility>
#include <vector>
#include <ranges>

namespace dolfinx::fem
{

namespace petsc
{	

template <std::floating_point T>
Mat create_cuda_matrix(const Form<PetscScalar, T>& a)
{
  la::SparsityPattern pattern = fem::create_sparsity_pattern(a);
  pattern.finalize();
  return la::petsc::create_cuda_matrix(a.mesh()->comm(), pattern);
}

template<std::floating_point T>
Mat create_cuda_block_matrix(std::vector<std::vector<std::shared_ptr<CUDAForm<PetscScalar,T>>>>& forms)
{
  int rows = forms.size();
  int cols = (rows) ? forms[0].size() : 0;
  std::vector<std::vector<const Form<PetscScalar,T>*>> a(rows);
  std::vector<std::vector<std::unique_ptr<dolfinx::la::SparsityPattern>>> patterns(rows);
  std::shared_ptr<const mesh::Mesh<PetscScalar>> mesh;
  std::array<std::vector<std::pair<
                 std::reference_wrapper<const dolfinx::common::IndexMap>, int>>,
             2>
      maps;
  std::vector<std::shared_ptr<dolfinx::common::IndexMap>> restricted_row_maps(rows, nullptr);
  std::vector<std::shared_ptr<dolfinx::common::IndexMap>> restricted_col_maps(cols, nullptr);
  
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      if (auto cuda_form = forms[row][col]; cuda_form) {
       auto form = cuda_form->form();
       if (cuda_form->restricted()) {
          patterns[row].push_back(
            std::make_unique<dolfinx::la::SparsityPattern>(compute_restricted_sparsity_pattern(cuda_form))
          );
          if (!restricted_row_maps[row]) restricted_row_maps[row] = cuda_form->restriction_index_map(0);
          if (!restricted_col_maps[col]) restricted_col_maps[col] = cuda_form->restriction_index_map(1);
	  }
        else {
          patterns[row].push_back(std::make_unique<dolfinx::la::SparsityPattern>(
				  create_sparsity_pattern(*cuda_form->form())));
        }
        a[row].push_back(form);
        if (!mesh) mesh = form->mesh();
      }
      else {
        patterns[row].push_back(nullptr);
        a[row].push_back(nullptr);
      }
    }
  }

  std::array<std::vector<std::shared_ptr<const FunctionSpace<PetscScalar>>>, 2> V
      = fem::common_function_spaces(extract_function_spaces(a));
  std::array<std::vector<int>, 2> bs_dofs;
  std::array<std::vector<std::shared_ptr<dolfinx::common::IndexMap>>, 2> restricted_maps
      = {restricted_row_maps, restricted_col_maps};
  /// hmmm - I think it probably won't work if we pass the unrestricted
  /// index maps to the sparsity pattern constructor.  . .
  for (std::size_t d = 0; d < 2; ++d)
  {
    for (std::size_t i = 0; i < V[d].size(); i++)
    {
      auto& space = V[d][i];
      std::shared_ptr<const dolfinx::common::IndexMap> imap = (restricted_maps[d][i])
	      ? restricted_maps[d][i] : space->dofmap()->index_map;
        
      maps[d].emplace_back(*imap,
                           space->dofmap()->index_map_bs());
      // is dofmap bs is different than indexmap bs?
      bs_dofs[d].push_back(space->dofmap()->bs());
    }
  }

  // OK now figure out how to build a matrix with this. . .  
  std::vector<std::vector<const dolfinx::la::SparsityPattern*>> p(rows);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t col = 0; col < cols; ++col)
      p[row].push_back(patterns[row][col].get());

  la::SparsityPattern pattern(mesh->comm(), p, maps, bs_dofs);
  pattern.finalize();
  // more work will be needed here to get the proper local-to-global index maps
  return la::petsc::create_cuda_matrix(mesh->comm(), pattern); 
}

} // namespace petsc
} // namespace dolfinx::fem
