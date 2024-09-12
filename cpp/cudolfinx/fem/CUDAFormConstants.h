// Copyright (C) 2020 James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cudolfinx/common/CUDA.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/utils.h>
#include <cuda.h>

namespace dolfinx {
namespace fem {

/// A wrapper for a form constant with data that is stored in the
/// device memory of a CUDA device.
template <dolfinx::scalar T>
class CUDAFormConstants
{
public:

  /// Create an empty collection constant values
  CUDAFormConstants()
    : _form(nullptr)
    , _num_constant_values()
    , _dconstant_values(0)
  {
  }
  //-----------------------------------------------------------------------------
  /// Create a collection constant values from a given form
  ///
  /// @param[in] cuda_context A context for a CUDA device
  /// @param[in] form The variational form whose constants are used
  CUDAFormConstants(
    const CUDA::Context& cuda_context,
    const Form<T>* form)
    : _form(form)
    , _num_constant_values()
    , _dconstant_values(0)
  {
    CUresult cuda_err;
    const char * cuda_err_description;

    const std::vector<T>
      constant_values = pack_constants(*_form);

    // Allocate device-side storage for constant values
    _num_constant_values = constant_values.size();
    if (_num_constant_values > 0) {
      size_t dconstant_values_size =
        _num_constant_values * sizeof(T);
      cuda_err = cuMemAlloc(
        &_dconstant_values, dconstant_values_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemAlloc() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }

      // Copy constant values to device
      cuda_err = cuMemcpyHtoD(
        _dconstant_values, constant_values.data(), dconstant_values_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuMemFree(_dconstant_values);
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
    }
  }
  //-----------------------------------------------------------------------------
  /// Destructor
  ~CUDAFormConstants()
  {
    if (_dconstant_values)
      cuMemFree(_dconstant_values);
  }
  //-----------------------------------------------------------------------------
  /// Copy constructor
  /// @param[in] form_constant The object to be copied
  CUDAFormConstants(const CUDAFormConstants& form_constant) = delete;

  /// Move constructor
  /// @param[in] form_constant The object to be moved
  CUDAFormConstants(CUDAFormConstants&& constants)
    : _form(constants._form)
    , _num_constant_values(constants._num_constant_values)
    , _dconstant_values(constants._dconstant_values)
  {
    constants._form = nullptr;
    constants._num_constant_values = 0;
    constants._dconstant_values = 0;
  }
  //-----------------------------------------------------------------------------
  /// Assignment operator
  /// @param[in] form_constant Another CUDAFormConstants object
  CUDAFormConstants& operator=(const CUDAFormConstants& form_constant) = delete;

  /// Move assignment operator
  /// @param[in] form_constant Another CUDAFormConstants object
  CUDAFormConstants& operator=(CUDAFormConstants&& constants)
  {
    _form = constants._form;
    _num_constant_values = constants._num_constant_values;
    _dconstant_values = constants._dconstant_values;
    constants._form = nullptr;
    constants._num_constant_values = 0;
    constants._dconstant_values = 0;
    return *this;
  }
  //-----------------------------------------------------------------------------
  /// Get the number of constant values that the constant applies to
  int32_t num_constant_values() const { return _num_constant_values; }

  /// Get the constant values that the constant applies to
  CUdeviceptr constant_values() const { return _dconstant_values; }

  /// Update the constant values by copying values from host to device
  void update_constant_values() const
  {
    CUresult cuda_err;
    const char * cuda_err_description;

    // Pack constants into an array
    const std::vector<T>  
      constant_values = pack_constants(*_form);
    assert(_num_constant_values == constant_values.size());

    // Copy constant values to device
    if (_num_constant_values > 0) {
      size_t dconstant_values_size =
        _num_constant_values * sizeof(T);
      cuda_err = cuMemcpyHtoD(
        _dconstant_values, constant_values.data(), dconstant_values_size);
      if (cuda_err != CUDA_SUCCESS) {
        cuMemFree(_dconstant_values);
        cuGetErrorString(cuda_err, &cuda_err_description);
        throw std::runtime_error(
          "cuMemcpyHtoD() failed with " + std::string(cuda_err_description) +
          " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
      }
    }
  }
  //-----------------------------------------------------------------------------


private:
  // The form that the constant applies to
  const Form<T>* _form;

  /// The number of constant values
  int32_t _num_constant_values;

  /// The constant values
  CUdeviceptr _dconstant_values;
};

} // namespace fem
} // namespace dolfinx

