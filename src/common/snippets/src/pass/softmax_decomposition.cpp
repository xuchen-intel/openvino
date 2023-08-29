// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/softmax_decomposition.hpp"

#include "snippets/itt.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/core/rt_info.hpp"

namespace ov {
namespace snippets {
namespace pass {

SoftmaxDecomposition::SoftmaxDecomposition() {
    MATCHER_SCOPE(SoftmaxDecomposition);
    auto match_softmax = ov::pass::pattern::wrap_type<ov::op::v1::Softmax, ov::op::v8::Softmax>();
    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        using namespace lowered;
        auto m_softmax = m.get_match_root();
        Output<Node> input;

        if (transformation_callback(m_softmax)) {
            return false;
        }

        const auto& pshape = m_softmax->get_input_partial_shape(0);
        if (pshape.is_dynamic())
            return false;

        const auto shape = pshape.get_shape();
        const auto rank = shape.size();

        int64_t axis;
        if (const auto softmax_v8 = ov::as_type_ptr<ov::op::v8::Softmax>(m_softmax)) {
            input = softmax_v8->input_value(0);
            OPENVINO_SUPPRESS_DEPRECATED_START
            axis = ov::normalize_axis(m_softmax->get_friendly_name(), softmax_v8->get_axis(), rank);
            OPENVINO_SUPPRESS_DEPRECATED_END
        } else if (const auto softmax_v1 = ov::as_type_ptr<ov::op::v1::Softmax>(m_softmax)) {
            input = softmax_v1->input_value(0);
            axis = softmax_v1->get_axis();
        } else {
            return false;
        }

        const auto float_min_constant = uint32_t(0xff7fffff);
        const auto zero_constant = uint32_t(0x00000000);
        // ReduceMax
        const auto vector_buffer_max = std::make_shared<op::VectorBuffer>();
        const auto fill_max = std::make_shared<op::Fill>(vector_buffer_max, 0, float_min_constant);
        const auto max = std::make_shared<ov::op::v1::Maximum>(input, fill_max);
        const auto horizon_max = std::make_shared<op::HorizonMax>(max);
        const auto broadcast_horizon_max = std::make_shared<op::BroadcastMove>(horizon_max,
                                           horizon_max->get_input_partial_shape(0));
        // Sub + Exp
        const auto sub = std::make_shared<ov::op::v1::Subtract>(input, broadcast_horizon_max);
        const auto exp = std::make_shared<ov::op::v0::Exp>(sub);
        // ReduceSum
        const auto vector_buffer_sum = std::make_shared<op::VectorBuffer>();
        const auto fill_sum = std::make_shared<op::Fill>(vector_buffer_sum, 0, zero_constant);
        const auto sum = std::make_shared<ov::op::v1::Add>(exp, fill_sum);
        const auto horizon_sum = std::make_shared<op::HorizonSum>(sum);
        // Div
        const auto pow = std::make_shared<op::PowerStatic>(horizon_sum, -1.f);
        const auto broadcast_pow = std::make_shared<op::BroadcastMove>(pow, horizon_sum->get_input_partial_shape(0));
        const auto mul = std::make_shared<ov::op::v1::Multiply>(exp, broadcast_pow);

        replace_node(m_softmax, mul);
        copy_runtime_info(m_softmax, {vector_buffer_max, fill_max, max, horizon_max, broadcast_horizon_max,
                                      sub, exp, vector_buffer_sum, fill_sum, sum, horizon_sum,
                                      pow, broadcast_pow, mul});
        mul->set_friendly_name(m_softmax->get_friendly_name());

        OPENVINO_ASSERT(axis < static_cast<int64_t>(rank), "Softmax has incorrect axis");
        std::vector<size_t> subtensor(rank, 1);
        for (size_t i = axis; i < rank; ++i)
            subtensor[i] = PortDescriptor::ServiceDimensions::FULL_DIM;

        // Set port descriptors for ReduceMax
        PortDescriptorUtils::set_port_descriptor_ptr(vector_buffer_max->input(0), std::make_shared<PortDescriptor>(vector_buffer_max->input(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(vector_buffer_max->output(0), std::make_shared<PortDescriptor>(vector_buffer_max->output(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(fill_max->input(0), std::make_shared<PortDescriptor>(fill_max->input(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(fill_max->output(0), std::make_shared<PortDescriptor>(fill_max->output(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(max->input(0), std::make_shared<PortDescriptor>(max->input(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(max->output(0), std::make_shared<PortDescriptor>(max->output(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(horizon_max->input(0), std::make_shared<PortDescriptor>(horizon_max->input(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(horizon_max->output(0), std::make_shared<PortDescriptor>(horizon_max->output(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(broadcast_horizon_max->input(0),
                    std::make_shared<PortDescriptor>(broadcast_horizon_max->input(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(broadcast_horizon_max->output(0),
                    std::make_shared<PortDescriptor>(broadcast_horizon_max->output(0), subtensor));
        // Set port descriptors for Sub + Exp
        PortDescriptorUtils::set_port_descriptor_ptr(sub->input(0), std::make_shared<PortDescriptor>(sub->input(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(sub->output(0), std::make_shared<PortDescriptor>(sub->output(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(exp->input(0), std::make_shared<PortDescriptor>(exp->input(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(exp->output(0), std::make_shared<PortDescriptor>(exp->output(0), subtensor));
        // Set port descriptors for ReduceSum
        PortDescriptorUtils::set_port_descriptor_ptr(vector_buffer_sum->input(0), std::make_shared<PortDescriptor>(vector_buffer_sum->input(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(vector_buffer_sum->output(0), std::make_shared<PortDescriptor>(vector_buffer_sum->output(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(fill_sum->input(0), std::make_shared<PortDescriptor>(fill_sum->input(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(fill_sum->output(0), std::make_shared<PortDescriptor>(fill_sum->output(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(sum->input(0), std::make_shared<PortDescriptor>(sum->input(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(sum->output(0), std::make_shared<PortDescriptor>(sum->output(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(horizon_sum->input(0), std::make_shared<PortDescriptor>(horizon_sum->input(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(horizon_sum->output(0), std::make_shared<PortDescriptor>(horizon_sum->output(0), subtensor));
        // Set port descriptors for Div
        PortDescriptorUtils::set_port_descriptor_ptr(pow->input(0), std::make_shared<PortDescriptor>(pow->input(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(pow->output(0), std::make_shared<PortDescriptor>(pow->output(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(broadcast_pow->input(0), std::make_shared<PortDescriptor>(broadcast_pow->input(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(broadcast_pow->output(0), std::make_shared<PortDescriptor>(broadcast_pow->output(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(mul->input(0), std::make_shared<PortDescriptor>(mul->input(0), subtensor));
        PortDescriptorUtils::set_port_descriptor_ptr(mul->output(0), std::make_shared<PortDescriptor>(mul->output(0), subtensor));

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(match_softmax, matcher_name);
    register_matcher(m, callback);
}

}  // namespace pass
}  // namespace snippets
}  // namespace ov
