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
 * @interface UnrollLoops
 * @brief Unroll loops containing Eltwise nodes.
 * @ingroup snippets
 */
class UnrollLoops : public Pass {
public:
    OPENVINO_RTTI("Unroll_loops", "Pass")
    bool run(LinearIR& linear_ir) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
