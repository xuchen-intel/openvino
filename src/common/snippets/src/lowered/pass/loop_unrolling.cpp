// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/loop_unrolling.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool LoopUnrolling::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::LoopUnrolling")

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov