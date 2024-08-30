// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later


#pragma once

#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/DirichletBC.h>

#if defined(HAS_CUDA_TOOLKIT)
#include <cudolfinx/common/CUDA.h>
#include <cudolfinx/common/CUDAStore.h>
#include <cudolfinx/fem/CUDADirichletBC.h>
#include <cudolfinx/fem/CUDADofMap.h>
#include <cudolfinx/fem/CUDAFormCoefficients.h>
#include <cudolfinx/fem/CUDAFormConstants.h>
#include <cudolfinx/fem/CUDAFormIntegral.h>
#include <cudolfinx/la/CUDAVector.h>
#endif

#if defined(HAS_CUDA_TOOLKIT)

namespace dolfinx {

namespace fem {

/// Consolidates all form classes into one
template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
class CUDAForm
{
public:
  /// Create GPU copies of data needed for assembly
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] form Pointer to the variational form
  CUDAForm(
    const CUDA::Context& cuda_context,
    Form<T,U>* form
  )
  : _coefficients(cuda_context, form, _dofmap_store)
  , _constants(cuda_context, form)
  , _form(form)
  , _compiled(false)
  {
    _coefficients = CUDAFormCoefficients<T,U>(cuda_context, form, _dofmap_store);  
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
      cuda_context, cujit_target, *_form, assembly_kernel_type,
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
  
  std::map<IntegralType, std::vector<CUDAFormIntegral<T,U>>>& integrals() {
    if (!_compiled) {
      throw std::runtime_error("Cannot access integrals for uncompiled cuda form!");
    }
    return _integrals;
  }

  CUDAFormCoefficients<T,U>& coefficients() { return _coefficients; }

  const CUDAFormConstants<T>& constants() { return _constants; }

  std::shared_ptr<const CUDADofMap> dofmap(size_t i) {return _dofmap_store.get_device_object(_form->function_spaces()[i]->dofmap().get()); }

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

private:
  // Cache of CUDADofMaps
  common::CUDAStore<DofMap, CUDADofMap> _dofmap_store;
  // Form coefficients
  CUDAFormCoefficients<T, U> _coefficients;
  // Form Constants
  CUDAFormConstants<T> _constants;
  std::map<IntegralType, std::vector<CUDAFormIntegral<T,U>>> _integrals;
  bool _compiled;
  Form<T,U>* _form;
};

} // end namespace fem

} // end namespace dolfinx

#endif
