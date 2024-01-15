// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_dnnl_emitters.hpp"
#include "emitters/utils.hpp"

using namespace Xbyak_aarch64;

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;

std::set<std::vector<element::Type>> jit_dnnl_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

jit_dnnl_emitter::jit_dnnl_emitter(jit_generator *host, cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node, ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {

    kind = dnnl_eltwise_tanh;
    alpha = 0.f;
    beta = 0.f;

    set_injector();
}

jit_dnnl_emitter::jit_dnnl_emitter(jit_generator *host, cpu_isa_t host_isa,
                                   dnnl_alg_kind_t algKind, float alpha, float beta,
                                   ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc), kind(algKind), alpha(alpha), beta(beta) {

    set_injector();
}

void jit_dnnl_emitter::set_injector() {
    if (host_isa_ != dnnl::impl::cpu::aarch64::asimd) {
        OPENVINO_THROW("jit_dnnl_emitter doesn't support ", host_isa_);
    }
    eltwise_injector_asimd = std::make_shared<dnnl::impl::cpu::aarch64::jit_uni_eltwise_injector_f32<dnnl::impl::cpu::aarch64::asimd>>(
                             h, kind, alpha, beta, 1.f);
}

void jit_dnnl_emitter::emit_code(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                                 const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const {
    emit_impl(in_idxs, out_idxs);
}

void jit_dnnl_emitter::emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_idxs, out_idxs);
    } else {
        OPENVINO_THROW("jit_dnnl_emitter doesn't support ", host_isa_);
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_dnnl_emitter::emit_isa(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;

    if (out_idxs[0] != in_idxs[0]) {
        auto in_idxs_u32 = convert_to_u32<size_t>(in_idxs);
        auto out_idxs_u32 = convert_to_u32<size_t>(out_idxs);
        h->mov(TReg(out_idxs_u32[0]).b, TReg(in_idxs_u32[0]).b);
    }
    eltwise_injector_asimd->compute_vector(out_idxs[0]);
}

void jit_dnnl_emitter::emit_data() const {
    eltwise_injector_asimd->prepare_table();
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov