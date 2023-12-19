// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_snippets_emitters.hpp"

#include <cpu/aarch64/jit_generator.hpp>

using namespace Xbyak_aarch64;

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

NopEmitter::NopEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : aarch64::jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

LoopBeginEmitter::LoopBeginEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : jit_emitter(h, isa) {
    loop_begin = ov::as_type_ptr<snippets::op::LoopBegin>(expr->get_node());
    if (!loop_begin)
        OPENVINO_THROW("LoopBeginEmitter invoked with invalid op argument");
    const auto& target_inputs = loop_begin->output(loop_begin->get_output_size() - 1).get_target_inputs();
    if (target_inputs.size() != 1)
        OPENVINO_THROW("LoopBeginEmitter invoked with invalid configuration: the last output must have exactly one "
                       "input attached");
    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(target_inputs.begin()->get_node()->shared_from_this());
    if (!loop_end)
        OPENVINO_THROW("LoopBeginEmitter invoked with invalid configuration: the last output must be LoopEnd");
    work_amount = loop_end->get_work_amount();
    evaluate_once = loop_end->get_evaluate_once();
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void LoopBeginEmitter::emit_code(const std::vector<size_t> &in,
                                 const std::vector<size_t> &out) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void LoopBeginEmitter::validate_arguments(const std::vector<size_t> &in,
                                          const std::vector<size_t> &out) const {
    if (!in.empty())
        OPENVINO_THROW("Invalid inputs size: expected 0 got ", in.size());
    if (out.size() != 1)
        OPENVINO_THROW("Invalid outputs size: expected 1 got ", out.size());
}

void LoopBeginEmitter::emit_impl(const std::vector<size_t>& in,
                                 const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in, out);
    } else {
        OPENVINO_THROW("LoopBegin emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void LoopBeginEmitter::emit_isa(const std::vector<size_t>& in,
                const std::vector<size_t>& out) const {
    XReg reg_work_amount = XReg(out[0]);

    // save previous register state (if there is an outer loop that uses this reg for example)
    if (!evaluate_once) {
        h->mov(reg_work_amount, work_amount);
    }
    // Note: loop address is not calculated at this point, so need to call calcJmpAddress() which is protected
    // or ready(), but they both set internal flags and that's not a desired way to use them.
    // So the most obvious WA is just to use current address manually
    loop_begin->begin_address = h->getCurr();
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov