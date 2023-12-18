// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

#define SNIPPETS_MAX_SNIPPETS_DIMS 12

struct jit_snippets_call_args {
    const void *src_ptrs[SNIPPETS_MAX_SNIPPETS_DIMS] = {};
    void *dst_ptrs[SNIPPETS_MAX_SNIPPETS_DIMS] = {};
    void *buffer_scratchpad_ptr = nullptr;
};

struct jit_snippets_compile_args {
    size_t parallel_executor_ndims = 1;
};

class NopEmitter : public jit_emitter {
public:
    NopEmitter(dnnl::impl::cpu::aarch64::jit_generator* h,
               dnnl::impl::cpu::aarch64::cpu_isa_t isa,
               const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override {}
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov