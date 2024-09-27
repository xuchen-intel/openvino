// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_brgemm_emitter.hpp"

#include "transformations/snippets/common/op/brgemm_cpu.hpp"
#include "emitters/utils.hpp"
#include "transformations/snippets/aarch64/op/brgemm_utils.hpp"

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using namespace ov::intel_cpu::brgemm_utils;

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
    h->mov(h->x16, reinterpret_cast<uint64_t>(BrgemmKernelExecutor::execute));
    internal_call_postamble();
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
