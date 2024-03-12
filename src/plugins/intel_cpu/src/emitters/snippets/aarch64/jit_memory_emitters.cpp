// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_memory_emitters.hpp"
#include "emitters/utils.hpp"

using namespace Xbyak_aarch64;

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

MemoryEmitter::MemoryEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : jit_emitter(h, isa) {
    const auto n = expr->get_node();
    src_prc = n->get_input_element_type(0);
    dst_prc = n->get_output_element_type(0);
}

jit_load_memory_emitter::jit_load_memory_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : MemoryEmitter(h, isa, expr) {
    if (src_prc != dst_prc)
        OV_CPU_JIT_EMITTER_THROW("jit_load_memory_emitter supports only equal input and output types but gets ",
                                 src_prc.get_type_name(),
                                 " and ",
                                 dst_prc.get_type_name());
    if (src_prc != ov::element::f32)
        OV_CPU_JIT_EMITTER_THROW("jit_load_memory_emitter only supports FP32 precision.");

    const auto load = std::dynamic_pointer_cast<snippets::op::Load>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(load != nullptr, "expects Load expression");
    count = load->get_count();
    byte_offset = load->get_offset();
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
    load_emitter.reset(new jit_load_emitter(h, isa, src_prc, dst_prc, count, byte_offset));
}

void jit_load_memory_emitter::emit_impl(const std::vector<size_t>& in,
                            const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in, out);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Load emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_load_memory_emitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    if (!load_emitter)
        OV_CPU_JIT_EMITTER_THROW("Load CPU emitter isn't initialized for jit_load_memory_emitter!");

    load_emitter->emit_code(in, out, aux_vec_idxs, aux_gpr_idxs);
}

void jit_load_memory_emitter::emit_data() const {
    load_emitter->emit_data();
}

jit_load_broadcast_emitter::jit_load_broadcast_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : MemoryEmitter(h, isa, expr) {
    if (src_prc != dst_prc)
        OV_CPU_JIT_EMITTER_THROW("BroadcastEmitters support only equal input and output types but gets ",
                                 src_prc.get_type_name(),
                                 " and ",
                                 dst_prc.get_type_name());
    if (src_prc != ov::element::f32)
        OV_CPU_JIT_EMITTER_THROW("jit_load_broadcast_emitter only supports FP32 precision.");

    const auto broadcast_load = std::dynamic_pointer_cast<snippets::op::BroadcastLoad>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(broadcast_load != nullptr, "expects BroadcastLoad expression");
    byte_offset = broadcast_load->get_offset();
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
}

void jit_load_broadcast_emitter::emit_impl(const std::vector<size_t>& in,
                                     const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in, out);
    } else {
        OV_CPU_JIT_EMITTER_THROW("BroadcastLoad emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_load_broadcast_emitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    XReg src = XReg(in[0]);
    TReg dst = TReg(out[0]);

    h->uni_ld1rw(dst.s, src, byte_offset);
}

jit_store_memory_emitter::jit_store_memory_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : MemoryEmitter(h, isa, expr) {
    if (src_prc != dst_prc)
        OV_CPU_JIT_EMITTER_THROW("jit_store_memory_emitter supports only equal input and output types but gets ",
                                 src_prc.get_type_name(),
                                 " and ",
                                 dst_prc.get_type_name());
    if (src_prc != ov::element::f32)
        OV_CPU_JIT_EMITTER_THROW("jit_store_memory_emitter only supports FP32 precision.");

    const auto store = ov::as_type_ptr<snippets::op::Store>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(store != nullptr, "expects Store expression");
    count = store->get_count();
    byte_offset = store->get_offset();
    in_out_type_ = emitter_in_out_map::vec_to_gpr;
    store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count, byte_offset));
}

void jit_store_memory_emitter::emit_impl(const std::vector<size_t>& in,
                             const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in, out);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Store emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_store_memory_emitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    if (!store_emitter)
        OV_CPU_JIT_EMITTER_THROW("Store CPU emitter isn't initialized for jit_store_memory_emitter!");

    store_emitter->emit_code(in, out, aux_vec_idxs, aux_gpr_idxs);
}

void jit_store_memory_emitter::emit_data() const {
    store_emitter->emit_data();
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
