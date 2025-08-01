// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/set_load_store_scalar.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/op/load.hpp"
#include "snippets/op/store.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::snippets::lowered::pass {

bool SetLoadStoreScalar::run([[maybe_unused]] LinearIR& linear_ir,
                             lowered::LinearIR::constExprIt begin,
                             lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::SetLoadStoreScalar")
    bool modified = false;
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto& expr = *expr_it;
        if (const auto load = ov::as_type_ptr<op::Load>(expr->get_node())) {
            const auto& desc = expr->get_input_port_descriptor(0);
            const auto& dim = desc->get_shape()[utils::get_input_dim_idx(desc->get_layout(), 0)];
            OPENVINO_ASSERT(!utils::is_dynamic_value(dim), "SetLoadStoreScalar expects static shapes");
            if (dim == 1) {
                load->set_count(1);
                modified = true;
            }
        } else if (const auto store = ov::as_type_ptr<op::Store>(expr->get_node())) {
            const auto& desc = expr->get_output_port_descriptor(0);
            const auto& dim = desc->get_shape()[utils::get_output_dim_idx(desc->get_layout(), 0)];
            OPENVINO_ASSERT(!utils::is_dynamic_value(dim), "SetLoadStoreScalar expects static shapes");
            if (dim == 1) {
                store->set_count(1);
                modified = true;
            }
        }
    }
    return modified;
}

}  // namespace ov::snippets::lowered::pass
