// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/softmax_decomposition.hpp"

#include "snippets/itt.hpp"
// #include "snippets/snippets_isa.hpp"
// #include "snippets/lowered/port_descriptor.hpp"
// #include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace snippets {
namespace pass {

SoftmaxDecomposition::SoftmaxDecomposition() {
    MATCHER_SCOPE(SoftmaxDecomposition);
}

}  // namespace pass
}  // namespace snippets
}  // namespace ov
