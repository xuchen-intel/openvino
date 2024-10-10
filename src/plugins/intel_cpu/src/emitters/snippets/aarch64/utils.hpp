// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/aarch64/jit_generator.hpp"
#include "snippets/emitter.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {
namespace utils {

inline static std::vector<Xbyak_aarch64::XReg> transform_idxs_to_regs(const std::vector<size_t>& idxs) {
    std::vector<Xbyak_aarch64::XReg> regs(idxs.size(), Xbyak_aarch64::XReg(0));
    std::transform(idxs.begin(), idxs.end(), regs.begin(), [](size_t idx){return Xbyak_aarch64::XReg(idx);});
    return regs;
}

inline static std::vector<size_t> transform_snippets_regs_to_idxs(const std::vector<snippets::Reg>& regs) {
    std::vector<size_t> idxs(regs.size());
    std::transform(regs.cbegin(), regs.cend(), idxs.begin(), [](const snippets::Reg& reg) { return reg.idx; });
    return idxs;
}

}   // namespace utils
}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
