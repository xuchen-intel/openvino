// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface SoftmaxDecomposition
 * @brief Decompose Softmax to a set of ReduceMax, Sub, Exp, ReduceSum and Div ops.
 * @ingroup snippets
 */
class SoftmaxDecomposition: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SoftmaxDecomposition", "0");
    SoftmaxDecomposition();
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
