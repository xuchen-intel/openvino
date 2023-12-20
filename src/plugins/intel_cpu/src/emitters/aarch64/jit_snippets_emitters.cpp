// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_snippets_emitters.hpp"

#include <cpu/aarch64/jit_generator.hpp>

#include "emitters/utils.hpp"

using namespace Xbyak_aarch64;

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

inline static void transform_idxs_to_regs(const std::vector<size_t>& idxs, std::vector<XReg>& regs) {
    regs.resize(idxs.size(), XReg(0));
    std::transform(idxs.begin(), idxs.end(), regs.begin(), [](size_t idx){return XReg(static_cast<uint32_t>(idx));});
}

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

LoopEndEmitter::LoopEndEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : jit_emitter(h, isa) {
    loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(expr->get_node());
    if (!loop_end)
        OPENVINO_THROW("LoopEndEmitter invoked with invalid op argument");
    loop_begin = loop_end->get_loop_begin();
    if (!loop_begin)
        OPENVINO_THROW("LoopEndEmitter invoked with invalid configuration: the last arg must be LoopBegin");
    num_inputs = loop_end->get_input_num();
    num_outputs = loop_end->get_output_num();
    wa_increment = static_cast<int64_t>(loop_end->get_increment());
    work_amount = static_cast<int64_t>(loop_end->get_work_amount());
    ptr_increments = loop_end->get_ptr_increments();
    finalization_offsets = loop_end->get_finalization_offsets();
    evaluate_once = loop_end->get_evaluate_once();
    io_data_size = loop_end->get_element_type_sizes();
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void LoopEndEmitter::emit_code(const std::vector<size_t> &in,
                               const std::vector<size_t> &out) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void LoopEndEmitter::validate_arguments(const std::vector<size_t> &in,
                                        const std::vector<size_t> &out) const {
    if (out.size() != num_outputs)
        OPENVINO_THROW("Invalid number of out arguments: expected ", num_outputs, " got ", out.size());
    if (in.size() != num_inputs)
        OPENVINO_THROW("Invalid number of in arguments: expected ", num_inputs , " got ", in.size());
    const auto io_size = num_inputs - 1;
    if (ptr_increments.size() != io_size)
        OPENVINO_THROW("Invalid ptr_increments size: expected ", io_size, " got ", ptr_increments.size());
    if (finalization_offsets.size() != io_size)
        OPENVINO_THROW("Invalid finalization_offsets size: expected: ", io_size, " got ", finalization_offsets.size());
}

void LoopEndEmitter::emit_impl(const std::vector<size_t>& in,
                               const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in, out);
    } else {
        OPENVINO_THROW("LoopEnd emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void LoopEndEmitter::emit_isa(const std::vector<size_t>& in,
                              const std::vector<size_t>& out) const {
    std::vector<size_t> data_ptr_reg_idxs;
    // the last input is actually a work_amount reg
    data_ptr_reg_idxs.reserve(num_inputs - 1);
    std::copy(in.begin(), in.end() - 1, std::back_inserter(data_ptr_reg_idxs));
    std::vector<XReg> data_ptr_regs;
    transform_idxs_to_regs(data_ptr_reg_idxs, data_ptr_regs);
    XReg reg_work_amount = XReg(in.back());
    if (!evaluate_once) {
        for (size_t idx = 0; idx < data_ptr_regs.size(); idx++) {
            if (ptr_increments[idx] != 0)
                h->add(data_ptr_regs[idx], data_ptr_regs[idx], ptr_increments[idx] * wa_increment * io_data_size[idx]);
        }
        h->sub(reg_work_amount, reg_work_amount, wa_increment);
        h->cmp(reg_work_amount, wa_increment);
        h->b(GE, reinterpret_cast<int64_t>(loop_begin->begin_address));
    }

    for (size_t idx = 0; idx < data_ptr_regs.size(); idx++) {
        if (finalization_offsets[idx] != 0)
            h->add(data_ptr_regs[idx], data_ptr_regs[idx], finalization_offsets[idx] * io_data_size[idx]);
    }
}

MemoryEmitter::MemoryEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : jit_emitter(h, isa) {
    const auto n = expr->get_node();
    src_prc = n->get_input_element_type(0);
    dst_prc = n->get_output_element_type(0);
}

LoadEmitter::LoadEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : MemoryEmitter(h, isa, expr) {
    if (src_prc != dst_prc)
        OPENVINO_THROW("LoadEmitter supports only equal input and output types but gets: ",
                       src_prc.get_type_name(),
                       " and ",
                       dst_prc.get_type_name());

    const auto load = std::dynamic_pointer_cast<snippets::op::Load>(expr->get_node());
    count = load->get_count();
    byte_offset = load->get_offset();
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
    load_emitter.reset(new jit_load_emitter(h, isa, src_prc, dst_prc, count));
}

void LoadEmitter::emit_impl(const std::vector<size_t>& in,
                            const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in, out);
    } else {
        OPENVINO_THROW("Load emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void LoadEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    if (!load_emitter)
        OPENVINO_THROW("Load CPU emitter isn't initialized for LoadEmitter!");
    load_emitter->emit_code({in[0], byte_offset}, {out[0]}, convert_to_size_t<uint32_t>(aux_vec_idxs),
                            convert_to_size_t<uint32_t>(aux_gpr_idxs));
}

void LoadEmitter::emit_data() const {
    load_emitter->emit_data();
}

StoreEmitter::StoreEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : MemoryEmitter(h, isa, expr) {
    if (src_prc != dst_prc)
        OPENVINO_THROW("StoreEmitter supports only equal input and output types but gets: ",
                       src_prc.get_type_name(),
                       " and ",
                       dst_prc.get_type_name());

    const auto store = ov::as_type_ptr<snippets::op::Store>(expr->get_node());
    count = store->get_count();
    byte_offset = store->get_offset();
    in_out_type_ = emitter_in_out_map::vec_to_gpr;
    store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count));
}

void StoreEmitter::emit_impl(const std::vector<size_t>& in,
                             const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in, out);
    } else {
        OPENVINO_THROW("Store emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void StoreEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    if (!store_emitter)
        OPENVINO_THROW("Store CPU emitter isn't initialized for StoreEmitter!");
    store_emitter->emit_code({in[0], byte_offset}, {out[0]}, convert_to_size_t<uint32_t>(aux_vec_idxs),
                             convert_to_size_t<uint32_t>(aux_gpr_idxs));
}

void StoreEmitter::emit_data() const {
    store_emitter->emit_data();
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
