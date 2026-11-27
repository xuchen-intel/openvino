// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "disable_bf16_comp_sin_cos_embedding.hpp"

#include <memory>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_cpu {

DisableBF16CompSinCosEmbedding::DisableBF16CompSinCosEmbedding() {
    MATCHER_SCOPE(DisableBF16CompSinCosEmbedding);
    using namespace ov::pass::pattern;

    // Match only the embedding's signature, so that unrelated Sin/Cos stay in BF16:
    // the outer product [N, 1] x [1, K] of the timestep and the frequencies, feeding both Sin and Cos.
    auto feeds_sin_and_cos = [](const ov::Output<ov::Node>& output) {
        bool has_sin = false;
        bool has_cos = false;
        for (const auto& input : output.get_target_inputs()) {
            has_sin |= ov::is_type<ov::op::v0::Sin>(input.get_node());
            has_cos |= ov::is_type<ov::op::v0::Cos>(input.get_node());
        }
        return has_sin && has_cos;
    };
    auto x_m = any_input(shape_matches("[?, 1]"));
    auto freq_m = wrap_type<ov::op::v0::Constant>(shape_matches("[1, ?]"));
    auto mul_m = wrap_type<ov::op::v1::Multiply>({x_m, freq_m}, feeds_sin_and_cos);
    auto sin_or_cos_m = wrap_type<ov::op::v0::Sin, ov::op::v0::Cos>({mul_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto sin_or_cos = pattern_map.at(sin_or_cos_m).get_node_shared_ptr();
        if (transformation_callback(sin_or_cos)) {
            return false;
        }

        // Keep exactly these nodes in their original precision, without spreading to the rest of the graph.
        // x must not be rounded to BF16 either, so mark the node computing it, together with the shape-only
        // ops, such as Unsqueeze, between that node and the Multiply.
        auto x = pattern_map.at(x_m).get_node_shared_ptr();
        while (ov::is_type<ov::op::v0::Unsqueeze>(x) || ov::is_type<ov::op::v1::Reshape>(x)) {
            ov::disable_conversion(x, ov::element::f16);
            x = x->get_input_node_shared_ptr(0);
        }
        if (!ov::is_type<ov::op::v0::Parameter>(x) && !ov::is_type<ov::op::v0::Constant>(x)) {
            ov::disable_conversion(x, ov::element::f16);
        }
        ov::disable_conversion(pattern_map.at(mul_m).get_node_shared_ptr(), ov::element::f16);
        ov::disable_conversion(sin_or_cos, ov::element::f16);
        return false;
    };

    auto m = std::make_shared<Matcher>(sin_or_cos_m, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_cpu
