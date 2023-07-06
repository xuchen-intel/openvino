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
        const auto& increment = loop_end->get_increment();
        // Ignore outer loops and tail loops
        if (increment < work_amount) {
            bool is_supported = true;
            // loop begin expr
            auto expr_copy_begin_it = expr_it;
            expr_it++;
            while ((*expr_it)->get_node() != loop_end) {
                const auto& node = (*expr_it)->get_node();
                if (ov::is_type<const snippets::op::MemoryAccess>(node)) {
                    expr_it++;
                    continue;
                }
                if (!is_supported_eltwise_node(node)) {
                    is_supported = false;
                    break;
                }
                expr_it++;
            }
            expr_it++;
            // the expr after loop end
            auto expr_copy_end_it = expr_it;

            if (is_supported) {
                modified = true;
                loop_end->set_unroll_loop(true);
                const size_t unroll_factor = std::min(default_unroll_factor, work_amount / increment);
                loop_end->set_unroll_factor(unroll_factor);
                loop_end->set_increment(unroll_factor * increment);
                auto loop_deep_copy = LinearIR::deep_copy_range(expr_copy_begin_it, expr_copy_end_it);
                auto to_erase = std::remove_if(loop_deep_copy.begin(), loop_deep_copy.end(),
                                [](const ExpressionPtr& expr) { return is_type<ov::op::v0::Parameter>(expr->get_node()) ||
                                                                       is_type<ov::op::v0::Result>(expr->get_node());});
                loop_deep_copy.erase(to_erase, loop_deep_copy.end());
                for (size_t i = 1; i < unroll_factor; i++) {
                    auto loop_insert = LinearIR::deep_copy_range(loop_deep_copy.begin(), loop_deep_copy.end());
                    for (auto expr_insert_it = loop_insert.begin(); expr_insert_it != loop_insert.end(); expr_insert_it++) {
                        // Reset offset for MemoryAccess nodes
                        if (const auto memory_access = std::dynamic_pointer_cast<ov::snippets::op::MemoryAccess>((*expr_insert_it)->get_node())) {
                            for (const auto in : memory_access->get_memory_access_input_ports()) {
                                const auto port = in.first;
                                memory_access->set_input_offset(memory_access->get_input_offset(port) + increment, port);
                            }
                            for (const auto out : memory_access->get_memory_access_output_ports()) {
                                const auto port = out.first;
                                memory_access->set_output_offset(memory_access->get_output_offset(port) + increment, port);
                            }
                        // Reset work_amount_increment and finalization_offsets for LoopEnd
                        } else if (const auto& loop_end_insert = ov::as_type_ptr<ov::snippets::op::LoopEnd>((*expr_insert_it)->get_node())) {
                            loop_end_insert->set_increment(unroll_factor * increment);
                            std::vector<int64_t> offsets = loop_end_insert->get_finalization_offsets();
                            std::transform(offsets.begin(), offsets.end(), offsets.begin(), [&increment](size_t offset) {return offset + increment;});
                            loop_end_insert->set_finalization_offsets(offsets);
                            loop_end_insert->has_outer_loop = loop_end->has_outer_loop;
                        }
                    }
                    linear_ir.insert(expr_it, loop_insert.begin(), loop_insert.end());
                }
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