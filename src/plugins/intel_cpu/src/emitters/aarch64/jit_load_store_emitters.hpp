// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

class jit_load_emitter : public jit_emitter {
public:
    jit_load_emitter(dnnl::impl::cpu::aarch64::jit_generator *host, dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                     ov::element::Type src_prc, ov::element::Type dst_prc, int load_num,
                     ov::element::Type exec_prc = ov::element::f32,
                     bool is_fill = false, std::string fill_value = "zero",
                     emitter_in_out_map in_out_type = emitter_in_out_map::gpr_to_vec);

    void emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const override;

    size_t get_inputs_count() const override { return 1; };

private:
    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const;

    std::string name_;
    int v_len_elt_;
    int load_num_;
    int load_size_;
    ov::element::Type src_prc_;
    ov::element::Type dst_prc_;
    bool is_fill_;
    std::string fill_value_;
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
