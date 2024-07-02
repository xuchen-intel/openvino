// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

class jit_convert_emitter : public jit_emitter {
public:
    jit_convert_emitter(dnnl::impl::cpu::aarch64::jit_generator *host, dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                        const std::shared_ptr<ov::Node>& n, ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_count() const override;

protected:
    void emit_data() const override;
    void validate_types() const;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void float2bfloat(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    ov::element::Type input_type;
    ov::element::Type output_type;

    const ov::element::TypeVector supported_types = {
            ov::element::f32,
            ov::element::i32,
            ov::element::bf16,
            ov::element::f16,
            ov::element::i8,
            ov::element::u8
    };
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
