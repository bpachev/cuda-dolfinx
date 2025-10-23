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
    std::vector<std::string>& tabulate_tensor_names,
    std::vector<std::string>& tabulate_tensor_sources,
    std::vector<int>& integral_tensor_indices
  )
  : _coefficients(cuda_context, form, _dofmap_store)
  , _constants(cuda_context, form)
  , _form(form)
  , _ufcx_form(ufcx_form)
  , _compiled(false)
  {
    _coefficients = CUDAFormCoefficients<T,U>(cuda_context, form, _dofmap_store);
    const int* integral_offsets = ufcx_form->form_integral_offsets;
    if (integral_offsets[3] != integral_tensor_indices.size()) {
      throw std::runtime_error("UFCx form has " + std::to_string(integral_offsets[3])
		      + " integrals, but only " + std::to_string(tabulate_tensor_names.size())
		      + " tabulate tensor sources provided to CUDAForm!"
		      );
    }
    for (int i = 0; i < 3; i++) {
      for (int offset = integral_offsets[i]; offset < integral_offsets[i+1]; offset++) {
        int id = ufcx_form->form_integral_ids[offset];
	int tensor_offset = integral_tensor_indices[offset];
        _cuda_integrals[i].insert({id, {tabulate_tensor_names[tensor_offset], tabulate_tensor_sources[tensor_offset]}});
      }
    } 
  }

  /// Compile form on GPU
  /// Under the hood, this creates the integrals
  void compile(
    const CUDA::Context& cuda_context,
    int32_t max_threads_per_block,
    int32_t min_blocks_per_multiprocessor,
    enum assembly_kernel_type assembly_kernel_type)
  {
    auto cujit_target = CUDA::get_cujit_target(cuda_context);
    _integrals = cuda_form_integrals(
      cuda_context, cujit_target, *_form, _cuda_integrals, assembly_kernel_type,
      max_threads_per_block, min_blocks_per_multiprocessor, false, NULL, false);
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
  // CUDA tabulate tensors 
  std::array<std::map<int, std::pair<std::string, std::string>>, 4> _cuda_integrals;
  bool _compiled;
  Form<T,U>* _form;
  ufcx_form* _ufcx_form;
};

} // end namespace fem

} // end namespace dolfinx
