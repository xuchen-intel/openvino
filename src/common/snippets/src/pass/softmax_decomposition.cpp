// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/softmax_decomposition.hpp"

#include "snippets/itt.hpp"
#include "snippets/snippets_isa.hpp"
// #include "snippets/lowered/port_descriptor.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/core/rt_info.hpp"

namespace ov {
namespace snippets {
namespace pass {

SoftmaxDecomposition::SoftmaxDecomposition() {
    MATCHER_SCOPE(SoftmaxDecomposition);
        auto softmax = ov::pass::pattern::wrap_type<ov::op::v1::Softmax, ov::op::v8::Softmax>();
        matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto m_softmax = m.get_match_root();
        // Output<Node> input;
        // int64_t softmax_axis;

        if (transformation_callback(m_softmax)) {
            return false;
        }

        // if (auto m_softmax_v1 = std::dynamic_pointer_cast<ov::op::v1::Softmax>(m_softmax)) {
        //     input = m_softmax_v1->input_value(0);
        //     softmax_axis = static_cast<int64_t>(m_softmax_v1->get_axis());
        // } else if (auto m_softmax_v8 = std::dynamic_pointer_cast<ov::op::v8::Softmax>(m_softmax)) {
        //     input = m_softmax_v8->input_value(0);
        //     softmax_axis = m_softmax_v8->get_axis();
        // } else {
        //     return false;
        // }

        const auto float_min_constant = uint32_t(0xff7fffff);
        const auto zero_constant = uint32_t(0x00000000);
        // ReduceMax
        const auto vector_buffer_max = std::make_shared<op::VectorBuffer>();
        const auto fill_max = std::make_shared<op::Fill>(vector_buffer_max, 0, float_min_constant);
        const auto max = std::make_shared<ov::op::v1::Maximum>(softmax->get_input_source_output(0), fill_max);
        const auto horizon_max = std::make_shared<op::HorizonMax>(max);
        const auto broadcast_horizon_max = std::make_shared<op::BroadcastMove>(horizon_max,
                                           horizon_max->get_input_partial_shape(0));
        const auto vector_buffer_sum = std::make_shared<op::VectorBuffer>();
        const auto fill_sum = std::make_shared<op::Fill>(vector_buffer_sum, 0, zero_constant);
        // Sub + Exp + ReduceSum
        const auto sub = std::make_shared<ov::op::v1::Subtract>(softmax->get_input_source_output(0), broadcast_horizon_max);
        const auto exp = std::make_shared<ov::op::v0::Exp>(sub);
        const auto sum = std::make_shared<ov::op::v1::Add>(exp, fill_sum);
        const auto horizon_sum = std::make_shared<op::HorizonSum>(sum);
        // Div
        const auto pow = std::make_shared<op::PowerStatic>(horizon_sum, -1.f);
        const auto broadcast_pow = std::make_shared<op::BroadcastMove>(pow, horizon_sum->get_input_partial_shape(0));
        const auto mul = std::make_shared<ov::op::v1::Multiply>(exp, broadcast_pow);

        replace_node(m_softmax, mul);
        copy_runtime_info(m_softmax, {vector_buffer_max, fill_max, max, horizon_max, broadcast_horizon_max,
                                      vector_buffer_sum, fill_sum, sub, exp, sum, horizon_sum,
                                      pow, broadcast_pow, mul});
        mul->set_friendly_name(m_softmax->get_friendly_name());
        return true;

        // auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {softmax_axis});
        // auto reduce_max = std::make_shared<ov::op::v1::ReduceMax>(input, axis, true);
        // auto sub = std::make_shared<ov::op::v1::Subtract>(input, reduce_max);
        // auto exp = std::make_shared<ov::op::v0::Exp>(sub);
        // auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(exp, axis, true);
        // auto div = std::make_shared<ov::op::v1::Divide>(exp, reduce_sum);

        // replace_node(m_softmax, div);
        // copy_runtime_info(m_softmax, {reduce_max, reduce_sum, sub, exp, div});
        // div->set_friendly_name(m_softmax->get_friendly_name());
        // return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(softmax, matcher_name);
    register_matcher(m, callback);
}

}  // namespace pass
}  // namespace snippets
}  // namespace ov
