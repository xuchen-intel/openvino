// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_dnnl_emitters.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

class jit_hswish_emitter : public jit_dnnl_emitter {
public:
    jit_hswish_emitter(dnnl::impl::cpu::aarch64::jit_generator *host, dnnl::impl::cpu::aarch64::cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& n,
                       ov::element::Type exec_prc = ov::element::f32) : jit_dnnl_emitter(host, host_isa, n, exec_prc) {
        // since v3.0 oneDNN has flexible version of hardswish, ov still uses the one with hardcoded alpha and beta
        kind = dnnl_eltwise_hardswish;
        alpha = 1.f / 6.f;
        beta = 0.5f;

        set_injector();
    }
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov