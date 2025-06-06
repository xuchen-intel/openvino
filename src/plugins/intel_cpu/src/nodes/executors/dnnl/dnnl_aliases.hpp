// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <oneapi/dnnl/dnnl.hpp>
#include <unordered_map>

namespace ov::intel_cpu {

using dnnl_primitive_args = std::unordered_map<int, dnnl::memory>;

}  // namespace ov::intel_cpu
