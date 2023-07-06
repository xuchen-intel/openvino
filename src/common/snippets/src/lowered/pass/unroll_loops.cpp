// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/unroll_loops.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool UnrollLoops::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::UnrollLoops")
    bool modified = false;
    // This is a default unrolling factor, given that currently register information is
    // unavailable in the stage of snippets common transformation
    constexpr size_t default_unroll_factor = 3;

    auto is_supported_eltwise_node = [](const std::shared_ptr<ov::Node>& node) {
        if (ov::is_type<const ov::op::v1::Maximum>(node) ||
            ov::is_type<const ov::op::v1::Subtract>(node) ||
            ov::is_type<const ov::op::v0::Exp>(node) ||
            ov::is_type<const ov::op::v1::Add>(node) ||
            ov::is_type<const ov::op::v1::Multiply>(node)) {
            return true;
        }
        return false;
    };

    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end();) {
        const auto& loop_begin = ov::as_type_ptr<ov::snippets::op::LoopBegin>((*expr_it)->get_node());
        if (!loop_begin) {
            expr_it++;
            continue;
        }
        const auto& loop_end = loop_begin->get_loop_end();
        const auto& work_amount = loop_end->get_work_amount();
        const auto& work_amount_increment = loop_end->get_increment();

        // Ignore outer loops and tail loops
        if (work_amount_increment < work_amount) {
            auto& loop_expr_it = expr_it;
            loop_expr_it++;
            bool is_supported = true;
            while ((*loop_expr_it)->get_node() != loop_end) {
                const auto& node = (*loop_expr_it)->get_node();
                if (ov::is_type<const snippets::op::MemoryAccess>(node)) {
                    loop_expr_it++;
                    continue;
                }
                if (!is_supported_eltwise_node(node)) {
                    is_supported = false;
                    break;
                }
                loop_expr_it++;
            }

            if (is_supported) {
                loop_end->set_unroll_loop(true);
                const size_t unroll_factor = std::min(default_unroll_factor, work_amount / work_amount_increment);
                loop_end->set_unroll_factor(unroll_factor);
                modified = true;
            }
        } else {
            expr_it++;
        }
    }

    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov