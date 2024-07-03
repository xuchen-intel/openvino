// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_conversion_emitters.hpp"
#include "emitters/utils.hpp"

using namespace dnnl::impl::cpu::aarch64;

namespace ov {
namespace intel_cpu {
namespace aarch64 {

jit_convert_emitter::jit_convert_emitter(jit_generator *host, cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {
    input_type = node->get_input_element_type(0);
    output_type = node->get_output_element_type(0);
}

void jit_convert_emitter::validate_types() const {
    OV_CPU_JIT_EMITTER_ASSERT(one_of(input_type, ov::element::f32, ov::element::i8),
                              "Unsupported input type: ", input_type.get_type_name());
    OV_CPU_JIT_EMITTER_ASSERT(one_of(output_type, ov::element::f32, ov::element::i8),
                              "Unsupported output type: ", output_type.get_type_name());
}

size_t jit_convert_emitter::get_inputs_count() const { return 1; }

void jit_convert_emitter::emit_data() const {
    jit_emitter::emit_data();
}

jit_convert_truncation_emitter::jit_convert_truncation_emitter(jit_generator *host, cpu_isa_t host_isa,
                                                               const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
        : jit_convert_emitter(host, host_isa, node, exec_prc) {
    prepare_table();
}

void jit_convert_truncation_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    validate_types();
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_convert_truncation_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
}

void jit_convert_truncation_emitter::register_table_entries() {
}

jit_convert_saturation_emitter::jit_convert_saturation_emitter(jit_generator *host, cpu_isa_t host_isa,
                                                               const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
    : jit_convert_emitter(host, host_isa, node, exec_prc) {
}

void jit_convert_saturation_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    validate_types();
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_convert_saturation_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
