// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of cuDOLFINX
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/common/IndexMap.h>
#include <cudolfinx/common/CUDA.h>
#include <cudolfinx/common/CUDAStore.h>
#include <cudolfinx/fem/CUDADirichletBC.h>
#include <cudolfinx/fem/CUDADofMap.h>
#include <cudolfinx/fem/CUDAFormCoefficients.h>
#include <cudolfinx/fem/CUDAFormConstants.h>
#include <cudolfinx/fem/CUDAFormIntegral.h>
#include <cudolfinx/la/CUDAVector.h>
#include <string>
#include <utility>
#include <ufcx.h>
#include <ranges>

namespace dolfinx {

namespace fem {

/// Consolidates all form classes into one
template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_t<T>>
class CUDAForm
{

public:
  /// Create GPU copies of data needed for assembly
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] form Pointer to the variational form
  CUDAForm(
    const CUDA::Context& cuda_context,
    Form<T,U>* form,
    ufcx_form* ufcx_form,
    const char * tabulate_tensor_source,
    char ** tabulate_tensor_names,
    std::string form_name
  )
  : _coefficients(cuda_context, form, _dofmap_store)
  , _constants(cuda_context, form)
  , _form(form)
  , _ufcx_form(ufcx_form)
  , _compiled(false)
  , _tabulate_tensor_source(tabulate_tensor_source)
  , _name(form_name)
  {
    _coefficients = CUDAFormCoefficients<T,U>(cuda_context, form, _dofmap_store);
    const int* integral_offsets = ufcx_form->form_integral_offsets;
    for (int i = 0; i < 3; i++) {
      for (int offset = integral_offsets[i]; offset < integral_offsets[i+1]; offset++) {
        int idx = offset - integral_offsets[i];
        _tabulate_tensor_names[i].insert({idx, std::string(tabulate_tensor_names[offset])});
      }
    }

    // initialize the CUDAFormIntegral objects
    // this mostly just intitializes the mesh entities for each integral
    // however there is an MPI component to determine the active ghost entities
    // and so the constructor needs to be called on all processes
    for (IntegralType integral_type : {IntegralType::cell, IntegralType::exterior_facet, IntegralType::interior_facet,
      IntegralType::vertex, IntegralType::ridge})

    {
      // TODO: add full mixed-topology support when it becomes more mature in dolfinx
      int num_integrals = form->num_integrals(integral_type, 0);
      if (num_integrals > 0) {
        std::vector<CUDAFormIntegral<T,U>>& cuda_integrals =
          _integrals[integral_type];
        for (int i = 0; i < num_integrals; i++)
          cuda_integrals.emplace_back(*form, integral_type, i); 
      }
    } 
  }

  /// Compile form on GPU
  /// Under the hood, this creates the integrals
  void compile(
    const CUDA::Context& cuda_context,
    int32_t max_threads_per_block,
    int32_t min_blocks_per_multiprocessor,
    std::string cachedir,
    enum assembly_kernel_type assembly_kernel_type)
  {
    auto cujit_target = CUDA::get_cujit_target(cuda_context);
      // Get the number of vertices and coordinates
    const mesh::Mesh<U>& mesh = *_form->mesh();
    std::int32_t num_vertices_per_cell = mesh::num_cell_vertices(mesh.geometry().cmap().cell_shape());
    //std::int32_t num_coordinates_per_vertex = mesh.geometry().dim();
    std::int32_t num_coordinates_per_vertex = 3;

    // Find the number of degrees of freedom per cell
    int32_t num_dofs_per_cell0 = 1;
    int32_t num_dofs_per_cell1 = 1;
    if (_form->rank() > 0) {
      const DofMap& dofmap0 = *_form->function_spaces()[0]->dofmap();
      num_dofs_per_cell0 = dofmap0.element_dof_layout().num_dofs() * dofmap0.element_dof_layout().block_size();
    }
    if (_form->rank() > 1) {
      const DofMap& dofmap1 = *_form->function_spaces()[1]->dofmap();
      num_dofs_per_cell1 = dofmap1.element_dof_layout().num_dofs() * dofmap1.element_dof_layout().block_size();
    }
 
    std::string assembly_src = _tabulate_tensor_source;
    // only add this function once
    if (_form->rank() == 2)
      assembly_src += cuda_kernel_binary_search() + "\n\n";

    std::map<IntegralType, std::vector<std::pair<std::string, std::string>>> kernel_names;

    for (auto& [integral_type, integrals_for_type] : _integrals) {
      auto names = _tabulate_tensor_names[static_cast<std::size_t>(integral_type)];
      for (int i = 0; i < integrals_for_type.size(); i++) {
        std::string kernel_name = "";
        auto it = names.find(i);
        if (it == names.end())
          throw std::runtime_error("No kernel for requested domain index.");

        std::string tabulate_tensor_function_name = it->second;
        auto [integral_assembly_src, assembly_kernel_name, lift_bc_kernel_name] = get_form_integral_kernel_src(
          _form->rank(),
          integral_type,
          tabulate_tensor_function_name,
          max_threads_per_block,
          min_blocks_per_multiprocessor,
          num_vertices_per_cell,
          num_coordinates_per_vertex,
          num_dofs_per_cell0,
          num_dofs_per_cell1,
          _form->coefficient_offsets().back(), // TODO fix this to handle 'active coefficients' properly
          assembly_kernel_type
        );
        // add assembly loop for this integral to the overall source
        assembly_src += integral_assembly_src;
        kernel_names[integral_type].emplace_back(assembly_kernel_name, lift_bc_kernel_name);
      }
    }

    _module = compile_form_assembly_module(
        cuda_context,
        cujit_target,
        assembly_src,
        _name,
        cachedir,
        false, // verbose
        false // debug
    );

    for (auto& [integral_type, integrals_for_type] : _integrals) {
      for (int i = 0; i < integrals_for_type.size(); i++) {
        auto [assembly_kernel_name, lift_bc_kernel_name] = kernel_names[integral_type][i];
        integrals_for_type[i].set_kernels(_module, assembly_kernel_name, lift_bc_kernel_name);
      }
    }

    _compiled = true;
  }

  /// Copy constructor
  CUDAForm(const CUDAForm& form) = delete;

  /// Move constructor
  CUDAForm(CUDAForm&& form) = default;

  /// Destructor
  virtual ~CUDAForm() = default;

  bool compiled() { return _compiled; }

  bool restricted() { return _restricted_dofmaps.size() > 0; }
  
  std::map<IntegralType, std::vector<CUDAFormIntegral<T,U>>>& integrals() {
    if (!_compiled) {
      throw std::runtime_error("Cannot access integrals for uncompiled cuda form!");
    }
    return _integrals;
  }

  CUDAFormCoefficients<T,U>& coefficients() { return _coefficients; }

  const CUDAFormConstants<T>& constants() { return _constants; }

  std::shared_ptr<const CUDADofMap> unrestricted_dofmap(size_t i) {
    if (i >= _form->function_spaces().size()) throw std::runtime_error("Dofmap index out of bounds!");
    return _dofmap_store.get_device_object(_form->function_spaces()[i]->dofmap().get());
  }

  std::shared_ptr<const CUDADofMap> dofmap(size_t i) {
    if (!restricted()) return unrestricted_dofmap(i);
    if (i >= _restricted_dofmaps.size()) throw std::runtime_error("Dofmap index out of bounds!");
    return _restricted_dofmaps[i];
  }

  Form<T,U>* form() { return _form; }

  CUDADirichletBC<T,U> bc(
    const CUDA::Context& cuda_context, size_t i,
    std::vector<std::shared_ptr<const DirichletBC<T,U>>> bcs)
  {
    return CUDADirichletBC<T,U>(cuda_context, *_form->function_spaces()[i], bcs);
  }

  /// Copy the coefficient and constant data to the device
  /// This can be necessary if either changes on the host
  void to_device(const CUDA::Context& cuda_context)
  {
    _coefficients.copy_coefficients_to_device(cuda_context);
    _constants.update_constant_values(); 
  }

  void set_restriction(
    std::vector<std::int32_t> offsets,
    std::vector<std::int32_t> ghost_offsets,
    std::vector<std::shared_ptr<std::map<std::int32_t, std::int32_t>>> restriction)
  {
    if (restriction.size() != _form->function_spaces().size()) {
      throw std::runtime_error("Number of restrictions must equal arity of form (1 for vector, 2 for matrix)!");
    }
    _restriction = restriction;
    if (_restricted_dofmaps.size()) {
      // need to update the restriction
      for (int i = 0; i < _restricted_dofmaps.size(); i++) {
        _restricted_dofmaps[i]->update(
          offsets[i],
          ghost_offsets[i],
          restriction[i].get()
        );
      } 
    }
    else {
      for (int i = 0; i < restriction.size(); i++) {
        _restricted_dofmaps.push_back(
          std::make_shared<CUDADofMap>(
            _form->function_spaces()[i]->dofmap().get(),
            offsets[i],
            ghost_offsets[i],
            restriction[i].get()
          )
	      );
      }
    }
  }

  const std::vector<std::shared_ptr<std::map<std::int32_t, std::int32_t>>> get_restriction()
  {
    return _restriction;
  }

  std::shared_ptr<dolfinx::common::IndexMap> restriction_index_map(size_t i) {
    std::vector<std::int32_t> restricted_inds;
    for (auto const& pair: *_restriction[i]) restricted_inds.push_back(pair.first);
    auto [sub_imap, inds] = dolfinx::common::create_sub_index_map(
        *_form->function_spaces()[0]->dofmap()->index_map,
        restricted_inds,
        dolfinx::common::IndexMapOrder::preserve, false
    ); 
    return std::make_shared<dolfinx::common::IndexMap>(std::move(sub_imap));
  }

private:
  // Cache of CUDADofMaps
  common::CUDAStore<DofMap, CUDADofMap> _dofmap_store;
  // Restricted dofmaps
  std::vector<std::shared_ptr<CUDADofMap>> _restricted_dofmaps;
  // Restriction
  std::vector<std::shared_ptr<std::map<std::int32_t, std::int32_t>>> _restriction;
  // Form coefficients
  CUDAFormCoefficients<T, U> _coefficients;
  // Form Constants
  CUDAFormConstants<T> _constants;
  // Compiled CUDA kernels
  std::map<IntegralType, std::vector<CUDAFormIntegral<T,U>>> _integrals;
  // CUDA tabulate tensor names
  std::array<std::map<int, std::string>, 4> _tabulate_tensor_names;
  // Source code with all tensor sources from FFcx
  std::string _tabulate_tensor_source;
  // Form name
  std::string _name;
  // Whether or not the form is compiled
  bool _compiled;
  // DOFLINx form object
  Form<T,U>* _form;
  // UFCx form object
  ufcx_form* _ufcx_form;
  // CUDA module
  CUDA::Module _module;
};

} // end namespace fem

} // end namespace dolfinx
