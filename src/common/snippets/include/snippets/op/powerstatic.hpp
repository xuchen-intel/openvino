// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <snippets/snippets_isa.hpp>

#include "openvino/op/op.hpp"

namespace ov {
namespace snippets {
namespace op {

/**
 * @interface PowerStatic
 * @brief Generated by Canonicalization for a spasical case of power innstruction which has constant power value
 * @ingroup snippets
 */
class PowerStatic : public ov::op::util::UnaryElementwiseArithmetic {
public:
    OPENVINO_OP("PowerStatic", "SnippetsOpset", ov::op::util::UnaryElementwiseArithmetic);

    PowerStatic() = default;
    PowerStatic(const Output<Node>& arg, float power) : UnaryElementwiseArithmetic(arg), power(power) {
        constructor_validate_and_infer_types();
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        check_new_args_count(this, new_args);
        return std::make_shared<PowerStatic>(new_args.at(0), power);
    }
    bool visit_attributes(AttributeVisitor& visitor) override {
        visitor.on_attribute("power", power);
        return true;
    }
    float get_power() const {
        return power;
    }

private:
    float power = 0;
};
}  // namespace op
}  // namespace snippets
}  // namespace ov
