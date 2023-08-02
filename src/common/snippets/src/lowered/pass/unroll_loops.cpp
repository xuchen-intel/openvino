// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/unroll_loops.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"
#include "snippets/utils.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool UnrollLoops::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::UnrollLoops")
    bool modified = false;
    bool assigned_vec_regs_initialized = false;
    std::set<size_t> assigned_vec_regs;
    std::set<PortConnectorPtr> port_connector_visited;
    // This is a default unrolling factor, given that currently register information is
    // unavailable in the stage of snippets common transformation
    constexpr size_t default_unroll_factor = 3;

    // Supported eltwise nodes decomposed from Softmax
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

    // Output vector regiters should be shared between unrolled loop bodies for nodes providing horizontal computation
    auto is_horizon_node = [&is_supported_eltwise_node](const std::shared_ptr<ov::Node>& node) {
        if (is_supported_eltwise_node(node) && (
            ov::is_type<const ov::op::v1::Maximum>(node) ||
            ov::is_type<const ov::op::v1::Add>(node))) {
            return true;
        }
        return false;
    };

    auto reassign_register = [](const std::vector<size_t>& vec_regs_unroll, const size_t& unroll_idx, size_t& reg){
        auto pos = static_cast<size_t>(find(vec_regs_unroll.begin(), vec_regs_unroll.end(), reg) - vec_regs_unroll.begin());
        // Only reassign registers that are not sharely used regs
        if (pos < vec_regs_unroll.size()) {
            // // Reassign vector registers cyclically based on unrolled body index
            // reg = vec_regs_unroll[(pos + unroll_idx) % vec_regs_unroll.size()];

            // Check#1
            // Hard code to increase all non-sharely vector register index by 4 * unroll_idx,
            // to check if assigning new registers can bring performance benefit
            reg = vec_regs_unroll[pos] + 4 * unroll_idx;
        }
        // Check#2
        // Increase all vector register index by 4 * unroll_idx, regardless of accuracy
        // hard code vector register #15 to be reserved for aux register of emitters
        // if (reg + 4 * unroll_idx < 15) {
        //     reg = reg + 4 * unroll_idx;
        // }
    };

    auto reassign_registers_for_expression = [&](const ExpressionPtr& expr,
        const std::vector<size_t>& vec_regs_unroll, const size_t& unroll_idx){
        auto rinfo = expr->get_reg_info();
        const auto& reg_type = m_reg_type_mapper(expr->get_node());
        if (ov::snippets::utils::one_of(reg_type, Generator::opRegType::vec2gpr, Generator::opRegType::vec2vec)) {
            for (size_t& reg : rinfo.first)
                reassign_register(vec_regs_unroll, unroll_idx, reg);
        }
        if (ov::snippets::utils::one_of(reg_type, Generator::opRegType::gpr2vec, Generator::opRegType::vec2vec)) {
            for (size_t& reg : rinfo.second)
                reassign_register(vec_regs_unroll, unroll_idx, reg);
        }
        expr->set_reg_info(rinfo);
    };

    // Reset offset for MemoryAccess nodes
    auto set_memory_access_offset = [](const std::shared_ptr<ov::Node>& node, const size_t& offset){
        if (const auto memory_access = std::dynamic_pointer_cast<ov::snippets::op::MemoryAccess>(node)) {
            for (const auto in : memory_access->get_memory_access_input_ports()) {
                const auto port = in.first;
                memory_access->set_input_offset(memory_access->get_input_offset(port) + offset, port);
            }
            for (const auto out : memory_access->get_memory_access_output_ports()) {
                const auto port = out.first;
                memory_access->set_output_offset(memory_access->get_output_offset(port) + offset, port);
            }
        }
    };

    // Initialize assigned_vec_regs
    auto init_assigned_vec_regs = [&]() {
        for (const auto& expr : linear_ir) {
            auto op = expr->get_node();
            auto reg_type = m_reg_type_mapper(op);
            auto rinfo = expr->get_reg_info();
            switch (reg_type) {
                case Generator::opRegType::gpr2gpr:
                    break;
                case Generator::opRegType::vec2gpr:
                    std::transform(rinfo.first.begin(), rinfo.first.end(), std::inserter(assigned_vec_regs, assigned_vec_regs.end()),
                    [](size_t reg){return reg;});
                    break;
                case Generator::opRegType::gpr2vec:
                    std::transform(rinfo.second.begin(), rinfo.second.end(), std::inserter(assigned_vec_regs, assigned_vec_regs.end()),
                    [](size_t reg){return reg;});
                    break;
                case Generator::opRegType::vec2vec:
                    std::transform(rinfo.first.begin(), rinfo.first.end(), std::inserter(assigned_vec_regs, assigned_vec_regs.end()),
                    [](size_t reg){return reg;});
                    std::transform(rinfo.second.begin(), rinfo.second.end(), std::inserter(assigned_vec_regs, assigned_vec_regs.end()),
                    [](size_t reg){return reg;});
                    break;
            }
        }
    };

    // Update the vector regs that can be cyclically used for unrolling loops
    auto update_vec_regs = [&is_horizon_node, &is_supported_eltwise_node](const LinearIR::container& loop_insert,
                           const std::set<size_t>& assigned_vec_regs, std::vector<size_t>& vec_regs_unroll){
        std::set<size_t> vec_regs = assigned_vec_regs;
        for (auto expr_insert_it = loop_insert.begin(); expr_insert_it != loop_insert.end(); expr_insert_it++) {
            const auto& expr = (*expr_insert_it);
            const auto& node = expr->get_node();
            if (is_horizon_node(node)) {
                auto rinfo = expr->get_reg_info();
                for (const auto& reg : rinfo.second)
                    vec_regs.erase(reg);
            }
            // In decomposed Softmax pattern, the binary Eltwise node right after Load has a sharely used regs
            // that should be excluded
            if (is_supported_eltwise_node(node)) {
                auto pre_expr_it = expr_insert_it;
                const auto& pre_expr = (*(--pre_expr_it));
                const auto& pre_node = pre_expr->get_node();
                if (ov::is_type<const snippets::op::Load>(pre_node)) {
                    auto pre_rinfo = pre_expr->get_reg_info();
                    if (pre_rinfo.second.size() != 1)
                        OPENVINO_THROW("snippets::op::Load must have only 1 register for output");
                    size_t load_out_reg = pre_rinfo.second[0];
                    std::vector<size_t> eltwise_in_regs = expr->get_reg_info().first;
                    eltwise_in_regs.erase(std::remove(eltwise_in_regs.begin(), eltwise_in_regs.end(), load_out_reg),
                                            eltwise_in_regs.end());
                    for (const auto& reg : eltwise_in_regs)
                        vec_regs.erase(reg);
                }
            }
        }
        vec_regs_unroll.assign(vec_regs.begin(), vec_regs.end());
    };

    auto update_vec_loop_expr = [](const LinearIR::container& loop_expr, std::vector<LinearIR::container>& vec_loop_expr){
        if (loop_expr.size() != vec_loop_expr.size())
            OPENVINO_THROW("loop_expr and vec_loop_expr should have same size!");
        auto vec_loop_expr_iter = vec_loop_expr.begin();
        for (auto expr_insert_it = loop_expr.begin(); expr_insert_it != loop_expr.end(); expr_insert_it++) {
            (*vec_loop_expr_iter).insert(vec_loop_expr_iter->end(), expr_insert_it, std::next(expr_insert_it));
            vec_loop_expr_iter++;
        }
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
            expr_it++;
            // The expr after LoopBegin
            auto expr_copy_begin_it = expr_it;
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
            // LoopEnd expr
            auto expr_copy_end_it = expr_it;

            //Unroll loops
            if (is_supported) {
                modified = true;

                // Initialize assigned_vec_regs only once
                if (!assigned_vec_regs_initialized) {
                    init_assigned_vec_regs();
                    assigned_vec_regs_initialized = true;
                }

                const size_t total_iters = work_amount / increment;
                const size_t unroll_factor = std::min(default_unroll_factor, total_iters);
                const size_t unroll_increment = unroll_factor * increment;
                auto loop_deep_copy = LinearIR::deep_copy_range(expr_copy_begin_it, expr_copy_end_it);
                auto to_erase = std::remove_if(loop_deep_copy.begin(), loop_deep_copy.end(),
                                [](const ExpressionPtr& expr) { return is_type<ov::op::v0::Parameter>(expr->get_node()) ||
                                                                       is_type<ov::op::v0::Result>(expr->get_node());});
                loop_deep_copy.erase(to_erase, loop_deep_copy.end());
                // Use std::vector<LinearIR::container> to maintain unrolled loops emitter by emitter during unrolling,
                // i.e. each original emitter and its duplications will be put in the same LinearIR::container
                std::vector<LinearIR::container> vec_loop_expr(loop_deep_copy.size());
                update_vec_loop_expr(loop_deep_copy, vec_loop_expr);
                // Insert loop remainder, including a copy of LoopBegin and LoopEnd
                bool has_loop_remainder = total_iters % unroll_factor;
                auto loop_remainder_insert = has_loop_remainder ?
                                             LinearIR::deep_copy_range(std::next(expr_copy_begin_it, -1), std::next(expr_copy_end_it)) :
                                             LinearIR::container();
                bool vec_regs_updated = false;
                std::vector<size_t> vec_regs_unroll;
                // Repeat loop body, excluding LoopBegin and LoopEnd
                for (size_t i = 1; i < unroll_factor; i++) {
                    auto loop_insert = LinearIR::deep_copy_range(loop_deep_copy.begin(), loop_deep_copy.end());
                    // Set offset for momory access nodes
                    for (auto expr_insert_it = loop_insert.begin(); expr_insert_it != loop_insert.end(); expr_insert_it++) {
                        set_memory_access_offset((*expr_insert_it)->get_node(), i * increment);
                    }
                    // Update vec_regs_unroll, to exclude sharely used regs
                    if (!vec_regs_updated) {
                        update_vec_regs(loop_insert, assigned_vec_regs, vec_regs_unroll);
                        vec_regs_updated = true;
                    }
                    // Reassign vector registers cyclically for unrolled loops
                    for (auto expr_insert_it = loop_insert.begin(); expr_insert_it != loop_insert.end(); expr_insert_it++) {
                        reassign_registers_for_expression(*expr_insert_it, vec_regs_unroll, i);
                    }

                    update_vec_loop_expr(loop_insert, vec_loop_expr);
                }
                // Replace original loop with unrolled loop
                for (auto expr_erase_it = expr_copy_begin_it; expr_erase_it != expr_copy_end_it;) {
                    expr_erase_it = linear_ir.erase(expr_erase_it);
                }
                for (const auto& loop_expr : vec_loop_expr) {
                    linear_ir.insert(expr_it, loop_expr.begin(), loop_expr.end());
                }
                loop_end->set_increment(unroll_increment);
                if (has_loop_remainder) {
                    // Finalization offset will only be applied in the final loop, which is in the loop remainder
                    loop_end->set_finalization_offsets(std::vector<int64_t>(loop_end->get_finalization_offsets().size(), 0));
                    // The expr after LoopEnd
                    expr_it++;
                    const size_t offset = work_amount / unroll_increment * unroll_increment;
                    auto remainder_loop_begin = linear_ir.insert(expr_it, loop_remainder_insert.begin(), loop_remainder_insert.end());
                    auto remainder_loop_end = ov::as_type_ptr<op::LoopBegin>((*remainder_loop_begin)->get_node())->get_loop_end();
                    remainder_loop_end->set_work_amount(work_amount - offset);
                    remainder_loop_end->set_increment(increment);
                } else {
                    expr_it++;
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
