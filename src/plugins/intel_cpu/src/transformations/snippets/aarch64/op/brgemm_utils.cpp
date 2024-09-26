// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_utils.hpp"

#include "emitters/utils.hpp"

using namespace dnnl::impl::cpu::aarch64;

namespace ov {
namespace intel_cpu {
namespace aarch64 {
namespace brgemm_utils {

cpu_isa_t get_primitive_isa() {
    auto isa = isa_undef;
#define SUPPORT(X, Y) if (mayiuse(X)) { isa = X; } else { Y }
#define SUPPORT_ONE(X, MESSAGE) SUPPORT(X, OV_CPU_JIT_EMITTER_THROW(MESSAGE);)
    SUPPORT_ONE(asimd, "Unsupported hardware configuration: brgemm requires at least asimd isa")
    return isa;
#undef SUPPORT_ONE
#undef SUPPORT
}

BRGEMM_TYPE get_brgemm_type() {
    return BRGEMM_TYPE::STAND_ALONE;
}

}   // namespace brgemm_utils
}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
