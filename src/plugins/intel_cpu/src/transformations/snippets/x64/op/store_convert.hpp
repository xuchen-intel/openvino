// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/op.hpp"
#include "snippets/op/store.hpp"

namespace ov::intel_cpu {

/**
 * @interface StoreConvertSaturation
 * @brief Fused operation to represent computations equal to consecutive Store and ConvertSaturation operations.
 *        The operation is used for peephole optimization during subgraph lowering.
 * @ingroup snippets
 */
class StoreConvertSaturation : public snippets::op::Store {
public:
    OPENVINO_OP("StoreConvertSaturation", "SnippetsOpset", snippets::op::Store);

    StoreConvertSaturation(const Output<Node>& x,
                           const ov::element::Type& destination_type,
                           size_t count = 1LU,
                           size_t offset = 0LU);
    StoreConvertSaturation() = default;

    ov::element::Type get_destination_type() const {
        return m_destination_type;
    }

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void validate_and_infer_types() override;

    bool has_evaluate() const override {
        return false;
    }

protected:
    ov::element::Type m_destination_type;
};

/**
 * @interface StoreConvertTruncation
 * @brief Fused operation to represent computations equal to consecutive Store and ConvertTruncation operations.
 *        The operation is used for peephole optimization during subgraph lowering.
 * @ingroup snippets
 */
class StoreConvertTruncation : public snippets::op::Store {
public:
    OPENVINO_OP("StoreConvertTruncation", "SnippetsOpset", snippets::op::Store);

    StoreConvertTruncation(const Output<Node>& x,
                           const ov::element::Type& destination_type,
                           size_t count = 1LU,
                           size_t offset = 0LU);
    StoreConvertTruncation() = default;

    ov::element::Type get_destination_type() const {
        return m_destination_type;
    }

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void validate_and_infer_types() override;

    bool has_evaluate() const override {
        return false;
    }

protected:
    ov::element::Type m_destination_type;
};

}  // namespace ov::intel_cpu
