// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

/**
 * @brief Keeps sinusoidal embeddings (e.g. diffusion timestep embeddings) in fp32
 *        when the CPU plugin infers in BF16.
 *
 *      x -> Multiply(x, Constant frequencies) -> Sin / Cos
 *
 *  The Multiply produces angles that can reach large magnitudes (FLUX feeds
 *  timestep * 1000, so the angles are up to ~1000 rad). BF16 has an 8-bit
 *  mantissa, so near 1000 its spacing is 4: rounding the angle before Sin/Cos
 *  turns the high-frequency channels of the embedding into noise.
 *
 *  The matched nodes are marked with @ref ov::disable_conversion(node, f32, bf16);
 *  CPU's EnforceInferencePrecision then expands this into a continuous fp32
 *  island (including the upstream angle-preparation ops) during precision
 *  enforcement.
 */
class DisableBF16CompSinCosEmbedding : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableBF16CompSinCosEmbedding");
    DisableBF16CompSinCosEmbedding();
};

}  // namespace ov::intel_cpu
