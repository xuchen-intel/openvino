// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_brgemm_emitter.hpp"

#include "transformations/snippets/common/op/brgemm_cpu.hpp"
#include "transformations/snippets/aarch64/op/brgemm_utils.hpp"
#include "emitters/utils.hpp"
#include "utils.hpp"
#include "cpu/aarch64/xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_adr.h"

using namespace Xbyak_aarch64;
using namespace ov::intel_cpu::brgemm_utils;

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

jit_brgemm_emitter::jit_brgemm_emitter(jit_generator* h, cpu_isa_t isa,
                                       const ExpressionPtr& expr,
                                       const snippets::KernelExecutorTablePtr& kernel_table,
                                       const ov::intel_cpu::MultiCacheWeakPtr& compiled_kernel_cache) :
                                       jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto& brgemm_node = as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node());
    const auto& brg0Prc = brgemm_node->get_input_element_type(0);
    const auto& brg1Prc = brgemm_node->get_input_element_type(1);
    const auto brgemm_type = brgemm_node->get_type();
    BrgemmKernelConfig kernel_config(brg0Prc, brg1Prc, brgemm_utils::get_primitive_isa());
    m_kernel_executor = kernel_table->register_kernel<BrgemmKernelExecutor>(expr,
                                                                            compiled_kernel_cache,
                                                                            kernel_config);
    OV_CPU_JIT_EMITTER_ASSERT(!snippets::utils::is_dynamic_vdims(expr->get_input_port_descriptor(0)->get_shape()) &&
                              !snippets::utils::is_dynamic_vdims(expr->get_input_port_descriptor(1)->get_shape()),
                              "Jit emitter is called when the shapes are unknown");
    auto get_cluster_id = [](const snippets::lowered::ExpressionPort& p) {
        if (const auto buffer = ov::as_type_ptr<ov::snippets::op::IntermediateMemoryBuffer>(p.get_expr()->get_node()))
            return buffer->get_cluster_id();
        else
            return SIZE_MAX;
    };
    m_memory_offsets = {brgemm_node->get_offset_a(), brgemm_node->get_offset_b(), brgemm_node->get_offset_c()};
    if (with_scratchpad(brgemm_type))
        m_memory_offsets.push_back(brgemm_node->get_offset_scratch());

    m_buffer_ids.assign(m_memory_offsets.size(), SIZE_MAX);
    for (size_t i = 0; i < m_memory_offsets.size(); i++) {
         if (snippets::utils::is_dynamic_value(m_memory_offsets[i])) {
             switch (i) {
                 case 0:
                 case 1:
                     m_buffer_ids[i] = get_cluster_id(expr->get_input_port_connector(i)->get_source());
                     break;
                 case 2:
                     for (const auto& child : expr->get_output_port_connector(0)->get_consumers())
                         if (!ov::is_type<snippets::op::LoopEnd>(child.get_expr()->get_node()))
                             m_buffer_ids[i] = get_cluster_id(child);
             }
             OV_CPU_JIT_EMITTER_ASSERT(m_buffer_ids[i] != SIZE_MAX, "Dynamic offset requires a valid buffer ID");
         }
    }
}

std::set<std::vector<element::Type>> jit_brgemm_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

void jit_brgemm_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT(m_memory_offsets.size() == in.size() + 1 && (out.size() == 1),
                              "expects 3 inputs if there are compensations/wsp");
}

void jit_brgemm_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    std::cout << "###### jit_brgemm_emitter::emit_impl" << std::endl;
    validate_arguments(in, out);
    std::vector<size_t> mem_ptrs_idxs{in[0], in[1], out[0]};
    if (in.size() > 2)
        mem_ptrs_idxs.emplace_back(in[2]);
    emit_brgemm_kernel_call(mem_ptrs_idxs, m_memory_offsets);
}

void jit_brgemm_emitter::emit_brgemm_kernel_call(const std::vector<size_t>& mem_ptrs_idxs, const std::vector<size_t>& mem_offsets) const {
    internal_call_preamble();
    auto reserved_stack_size = sizeof(BrgemmKernelExecutor::call_args);
    // Reserve memory on the stack
    h->sub(h->sp, h->sp, reserved_stack_size);
    h->mov(h->X_DEFAULT_ADDR, h->sp);

    XReg abi_param1 = XReg(0);
    auto write_addr_on_stack = [&](size_t arg_offset, XReg addr, size_t addr_offset, size_t buffer_id) {
        const auto stack_frame = pre_ptr(h->X_DEFAULT_ADDR, arg_offset);
        h->mov(h->X_TMP_0, addr);
        if (snippets::utils::is_dynamic_value(addr_offset))
            h->add_imm(h->X_TMP_0, abi_param1, static_cast<int32_t>(GET_OFF(buffer_offsets) + buffer_id * sizeof(size_t)), h->X_TMP_1);
        else if (addr_offset != 0)
            h->add_imm(h->X_TMP_0, abi_param1, static_cast<int32_t>(addr_offset), h->X_TMP_1);
        h->str(h->X_TMP_0, stack_frame);
    };
    const std::vector<size_t> brgemm_args_offsets {GET_OFF_BRGEMM_ARGS(A), GET_OFF_BRGEMM_ARGS(B), GET_OFF_BRGEMM_ARGS(C),
                                                   GET_OFF_BRGEMM_ARGS(scratch)};
    const auto& mem_ptrs = utils::transform_idxs_to_regs(mem_ptrs_idxs);
    for (size_t i = 0; i < mem_ptrs.size(); i++)
        write_addr_on_stack(brgemm_args_offsets[i], mem_ptrs[i], mem_offsets[i], m_buffer_ids[i]);

    // No scratchpad => need to write nullptr manually
    if (mem_ptrs.size() < 4) {
        h->mov(h->X_TMP_0, reinterpret_cast<uintptr_t>(nullptr));
        h->str(h->X_TMP_0, pre_ptr(h->X_DEFAULT_ADDR, brgemm_args_offsets.back()));
    }

    XReg abi_param2 = XReg(1);
    h->mov(abi_param1, reinterpret_cast<uintptr_t>(m_kernel_executor.get()));
    h->mov(abi_param2, h->sp);
    h->mov(h->x16, reinterpret_cast<uint64_t>(BrgemmKernelExecutor::execute));

    internal_call_rsp_align();
    h->blr(h->x16);
    internal_call_rsp_restore();

    h->add(h->sp, h->sp, reserved_stack_size);
    internal_call_postamble();
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
