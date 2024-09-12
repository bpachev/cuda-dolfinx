// Copyright (C) 2024 Benjamin Pachev, James D. Trotter
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once
#include <cudolfinx/common/CUDA.h>
#include <map>

namespace dolfinx::common
{
/// @brief This class represents an abstract mapping between host-side
/// and device-side objects. Its purpose is to prevent creation of duplicate
/// copies of host-side objects on the device.

template <class H, class D>
class CUDAStore
{
public:
  
  /// @brief Empty constructor
  CUDAStore()
  {
  }

  /// @brief Return stored device object, or update accordingly
  /// @param[in] host_object Shared pointer to the host-side object
  std::shared_ptr<D> get_device_object(const H* host_object) {
    auto it = _map.find(host_object);
    if (it != _map.end()) return it->second;
    auto device_object = std::make_shared<D>(host_object);
    _map[host_object] = device_object;
    return device_object;
  }

private:
  std::map<const H*, std::shared_ptr<D>> _map;  
};
}

