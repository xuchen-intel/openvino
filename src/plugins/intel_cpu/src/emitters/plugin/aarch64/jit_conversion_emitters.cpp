// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_conversion_emitters.hpp"
#include "emitters/utils.hpp"

using namespace dnnl::impl::cpu::aarch64;
using namespace Xbyak_aarch64;

namespace ov {
namespace intel_cpu {
namespace aarch64 {

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_convert_emitter::cvt_f16_to_f32(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_convert_emitter::cvt_f32_to_f16(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
    h->fcvtn(dst.h4, src.s4);
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_convert_emitter::cvt_f32_to_i32(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
    h->fcvtzs(dst.s, src.s);
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_convert_emitter::cvt_i32_to_f32(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_convert_emitter::cvt_i32_to_byte(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs, bool is_saturation) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_idxs[0]);
    TReg dst = TReg(out_idxs[0]);
    if (is_saturation) {
        if (output_type.is_signed()) {
        } else {
        }
    } else {
        h->xtn(dst.h4, src.s4);
        h->xtn(dst.b8, dst.h8);
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_convert_emitter::cvt_byte_to_i32(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs, bool is_saturation) const {
}

jit_convert_emitter::jit_convert_emitter(jit_generator *host, cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {
    input_type = node->get_input_element_type(0);
    output_type = node->get_output_element_type(0);
}

void jit_convert_emitter::validate_types() const {
    OV_CPU_JIT_EMITTER_ASSERT(one_of(input_type, ov::element::f32, ov::element::f16, ov::element::i8, ov::element::u8),
                              "Unsupported input type: ", input_type.get_type_name());
    OV_CPU_JIT_EMITTER_ASSERT(one_of(output_type, ov::element::f32, ov::element::f16, ov::element::i8, ov::element::u8),
                              "Unsupported output type: ", output_type.get_type_name());
}

size_t jit_convert_emitter::get_inputs_count() const { return 1; }

void jit_convert_emitter::emit_data() const {
    jit_emitter::emit_data();
}

jit_convert_truncation_emitter::jit_convert_truncation_emitter(jit_generator *host, cpu_isa_t host_isa,
                                                               const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
        : jit_convert_emitter(host, host_isa, node, exec_prc) {
}

void jit_convert_truncation_emitter::emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    validate_types();
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_idxs, out_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_convert_truncation_emitter::emit_isa(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    switch (output_type) {
        case ov::element::f32:
            switch (input_type) {
                case ov::element::f32:
                    break;
                case ov::element::i32:
                    cvt_i32_to_f32<isa>(in_idxs, out_idxs);
                    break;
                case ov::element::f16:
                    cvt_f16_to_f32<isa>(in_idxs, out_idxs);
                    break;
                case ov::element::i8:
                case ov::element::u8:
                    cvt_byte_to_i32<isa>(in_idxs, out_idxs, false);
                    cvt_i32_to_f32<isa>(out_idxs, out_idxs);
                    break;
                default:
                    OV_CPU_JIT_EMITTER_THROW("Unsupported input type: ", input_type.get_type_name());
            }
            break;
        case ov::element::i32:
            switch (input_type) {
                case ov::element::f32:
                    cvt_f32_to_i32<isa>(in_idxs, out_idxs);
                    break;
                case ov::element::i32:
                    break;
                case ov::element::f16:
                    cvt_f16_to_f32<isa>(in_idxs, out_idxs);
                    cvt_f32_to_i32<isa>(out_idxs, out_idxs);
                    break;
                case ov::element::i8:
                case ov::element::u8:
                    cvt_byte_to_i32<isa>(in_idxs, out_idxs, false);
                    break;
                default:
                    OV_CPU_JIT_EMITTER_THROW("Unsupported input type: ", input_type.get_type_name());
            }
            break;
        case ov::element::f16:
            switch (input_type) {
                case ov::element::f32:
                    cvt_f32_to_f16<isa>(in_idxs, out_idxs);
                    break;
                case ov::element::i32:
                    cvt_i32_to_f32<isa>(in_idxs, out_idxs);
                    cvt_f32_to_f16<isa>(out_idxs, out_idxs);
                    break;
                case ov::element::f16:
                    break;
                case ov::element::i8:
                case ov::element::u8:
                    cvt_byte_to_i32<isa>(in_idxs, out_idxs, false);
                    cvt_i32_to_f32<isa>(out_idxs, out_idxs);
                    cvt_f32_to_f16<isa>(out_idxs, out_idxs);
                    break;
                default:
                    OV_CPU_JIT_EMITTER_THROW("Unsupported input type: ", input_type.get_type_name());
            }
            break;
        case ov::element::i8:
        case ov::element::u8:
            switch (input_type) {
                case ov::element::f32:
                    cvt_f32_to_i32<isa>(in_idxs, out_idxs);
                    cvt_i32_to_byte<isa>(out_idxs, out_idxs, false);
                    break;
                case ov::element::i32:
                    cvt_i32_to_byte<isa>(in_idxs, out_idxs, false);
                    break;
                case ov::element::f16:
                    cvt_f16_to_f32<isa>(in_idxs, out_idxs);
                    cvt_f32_to_i32<isa>(out_idxs, out_idxs);
                    cvt_i32_to_byte<isa>(out_idxs, out_idxs, false);
                    break;
                case ov::element::i8:
                case ov::element::u8:
                    break;
                default:
                    OV_CPU_JIT_EMITTER_THROW("Unsupported input type: ", input_type.get_type_name());
            }
            break;
        default:
            OV_CPU_JIT_EMITTER_THROW("Unsupported output type: ", output_type.get_type_name());
    }
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
