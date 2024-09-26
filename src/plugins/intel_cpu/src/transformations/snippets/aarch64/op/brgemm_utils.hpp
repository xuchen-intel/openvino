// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "transformations/snippets/common/op/brgemm_common_utils.hpp"
#include "cpu/aarch64/cpu_isa_traits.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {
namespace brgemm_utils {

dnnl::impl::cpu::aarch64::cpu_isa_t get_primitive_isa();

BRGEMM_TYPE get_brgemm_type();

}   // namespace brgemm_utils
}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
