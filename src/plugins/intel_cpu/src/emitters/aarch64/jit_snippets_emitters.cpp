// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_snippets_emitters.hpp"

#include <cpu/aarch64/jit_generator.hpp>

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

NopEmitter::NopEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : aarch64::jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov