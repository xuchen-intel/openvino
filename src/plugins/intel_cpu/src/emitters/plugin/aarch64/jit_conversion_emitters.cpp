// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_conversion_emitters.hpp"

using namespace dnnl::impl::cpu::aarch64;

namespace ov {
namespace intel_cpu {
namespace aarch64 {

jit_convert_emitter::jit_convert_emitter(jit_generator *host, cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {
}

void jit_convert_emitter::validate_types() const {
}

size_t jit_convert_emitter::get_inputs_count() const { return 1; }

void jit_convert_emitter::emit_data() const {
}

template <cpu_isa_t isa>
void jit_convert_emitter::float2bfloat(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
}

jit_convert_truncation_emitter::jit_convert_truncation_emitter(jit_generator *host, cpu_isa_t host_isa,
                                                               const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
        : jit_convert_emitter(host, host_isa, node, exec_prc) {
}

bool jit_convert_truncation_emitter::is_i8_and_u8_case() const {
    return false;
}

void jit_convert_truncation_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
}

template <cpu_isa_t isa>
void jit_convert_truncation_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
}

void jit_convert_truncation_emitter::register_table_entries() {
}

template <cpu_isa_t isa>
void jit_convert_truncation_emitter::dword2int8(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
}

jit_convert_saturation_emitter::jit_convert_saturation_emitter(jit_generator *host, cpu_isa_t host_isa,
                                                               const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
    : jit_convert_emitter(host, host_isa, node, exec_prc) {
}

void jit_convert_saturation_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
}

template <cpu_isa_t isa>
void jit_convert_saturation_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
}

template <cpu_isa_t isa>
void jit_convert_saturation_emitter::dword2int8(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs, bool is_signed) const {
}

size_t jit_convert_saturation_emitter::get_aux_vecs_count() const { return 0; }

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
