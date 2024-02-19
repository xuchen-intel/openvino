// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/aarch64/jit_emitter.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

class LoopBeginEmitter : public jit_emitter {
public:
    LoopBeginEmitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                     dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                     const ov::snippets::lowered::ExpressionPtr& expr);

    void emit_code(const std::vector<size_t> &in,
                   const std::vector<size_t> &out) const;
    size_t get_inputs_count() const override {return 0;}

private:
    using jit_emitter::emit_code;
    void validate_arguments(const std::vector<size_t> &in,
                            const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;
    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in,
                  const std::vector<size_t>& out) const;

    std::shared_ptr<snippets::op::LoopBegin> loop_begin;
    bool evaluate_once = false;
    size_t work_amount = 0;
};

class LoopEndEmitter : public jit_emitter {
public:
    LoopEndEmitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                   dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                   const ov::snippets::lowered::ExpressionPtr& expr);

    void emit_code(const std::vector<size_t> &in,
                   const std::vector<size_t> &out) const;
    size_t get_inputs_count() const override {return 0;}

private:
    using jit_emitter::emit_code;
    void validate_arguments(const std::vector<size_t> &in,
                            const std::vector<size_t> &out) const override;

    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;
    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in,
                  const std::vector<size_t>& out) const;

    std::shared_ptr<snippets::op::LoopBegin> loop_begin;
    std::shared_ptr<snippets::op::LoopEnd> loop_end;

    size_t num_inputs = 0;
    size_t num_outputs = 0;
    // keep data_size int64_t to avoid conversion to size_t (and overflow) when multiplied by negative increments or offsets
    std::vector<int64_t> io_data_size {};
    int64_t wa_increment = 0;
    int64_t work_amount = 0;
    bool evaluate_once = false;
    std::vector<bool> is_incremented;
    std::vector<int64_t> ptr_increments;
    std::vector<int64_t> finalization_offsets;
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
