// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface LoopUnrolling
 * @brief Unroll loops containing Eltwise nodes.
 * @ingroup snippets
 */
class LoopUnrolling : public Pass {
public:
    OPENVINO_RTTI("LoopUnrolling", "Pass")
    bool run(LinearIR& linear_ir) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
