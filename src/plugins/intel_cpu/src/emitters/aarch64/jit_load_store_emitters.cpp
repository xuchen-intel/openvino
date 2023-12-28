// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_load_store_emitters.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;

jit_load_emitter::jit_load_emitter(dnnl::impl::cpu::aarch64::jit_generator *host, dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   ov::element::Type src_prc, ov::element::Type dst_prc, int load_num, int byte_offset,
                                   ov::element::Type exec_prc, bool is_fill, std::string fill_value, emitter_in_out_map in_out_type)
: jit_emitter(host, host_isa, exec_prc, in_out_type), name_("unknown"), load_num_(load_num), byte_offset_(byte_offset),
              src_prc_(src_prc), dst_prc_(dst_prc), is_fill_(is_fill), fill_value_(fill_value) {
    prepare_table();
    load_size_ = load_num * src_prc.size();
    v_len_elt_ = get_vec_length() / exec_prc.size();
}

void jit_load_emitter::emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_idxs, out_idxs);
    } else {
        OPENVINO_THROW("Load emitter in ", name_, " is performed on unsupported isa.");
    }
}

// todo: implement
template <cpu_isa_t isa>
void jit_load_emitter::emit_isa(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {}

jit_store_emitter::jit_store_emitter(dnnl::impl::cpu::aarch64::jit_generator *host, dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     ov::element::Type src_prc, ov::element::Type dst_prc, int store_num, int byte_offset,
                                     arithmetic_mode mode, ov::element::Type exec_prc, emitter_in_out_map in_out_type)
    : jit_emitter(host, host_isa, exec_prc, in_out_type), name_("unknown"), store_num_(store_num), byte_offset_(byte_offset),
                  src_prc_(src_prc), dst_prc_(dst_prc), mode_(mode) {
    prepare_table();
    v_len_elt_ = get_vec_length() / exec_prc.size();
    store_size_ = store_num * dst_prc.size();
}

void jit_store_emitter::emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_idxs, out_idxs);
    } else {
        OPENVINO_THROW("Store emitter in ", name_, " is performed on unsupported isa.");
    }
}

// todo: implement
template <cpu_isa_t isa>
void jit_store_emitter::emit_isa(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
