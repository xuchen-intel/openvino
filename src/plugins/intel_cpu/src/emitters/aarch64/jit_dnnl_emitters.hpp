// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/aarch64/jit_generator.hpp>
#include <cpu/aarch64/injectors/jit_uni_eltwise_injector.hpp>
#include "jit_emitter.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

class jit_dnnl_emitter : public jit_emitter {
public:
    void emit_code(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                   const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const override;
    void emit_data() const override;

    size_t get_inputs_count() const override {return 1;}
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

protected:
    jit_dnnl_emitter(dnnl::impl::cpu::aarch64::jit_generator *host, dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                     dnnl_alg_kind_t algKind, float inpAlpha, float inpBeta, ov::element::Type exec_prc = ov::element::f32);
    jit_dnnl_emitter(dnnl::impl::cpu::aarch64::jit_generator *host, dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                     const std::shared_ptr<ov::Node>& n, ov::element::Type exec_prc = ov::element::f32);

    void set_injector();

    dnnl_alg_kind_t kind {dnnl_alg_kind_undef};
    float alpha {0.f};
    float beta {0.f};

    std::shared_ptr<dnnl::impl::cpu::aarch64::jit_uni_eltwise_injector_f32<dnnl::impl::cpu::aarch64::asimd>> eltwise_injector_asimd;

private:
    void emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const override;
    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const;
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov