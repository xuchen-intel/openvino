// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_to_brgemm_cpu.hpp"

#include <cstddef>
#include <memory>
#include <vector>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/itt.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/op/rank_normalization.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"
#include "transformations/tpp/common/op/modifiers.hpp"

namespace ov::intel_cpu {

using namespace snippets::lowered;
using PortDescriptor = ov::snippets::modifier::MemoryAccess::PortDescriptor;

bool pass::BrgemmToBrgemmCPU::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(BrgemmToBrgemmCPU);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::BrgemmToBrgemmCPU")
    auto is_not_tpp = [](const Output<Node>& out) {
        return !std::dynamic_pointer_cast<const intel_cpu::tpp::modifier::TensorProcessingPrimitive>(
            out.get_node_shared_ptr());
    };
    auto m_brgemm = ov::pass::pattern::wrap_type<snippets::op::Brgemm>(is_not_tpp);
    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(m_brgemm);

    bool status = false;
    for (const auto& n : model->get_ordered_ops()) {
        if (!matcher->match(n)) {
            continue;
        }

        const auto node = matcher->get_match_root();
        const auto brgemm = ov::as_type_ptr<snippets::op::Brgemm>(node);
        const auto brgemm_plugin = ov::as_type_ptr<BrgemmCPU>(node);
        OPENVINO_ASSERT(brgemm && !brgemm_plugin, "BrgemmCPU cannot be in body before BrgemmToBrgemmCPU pass");
        const auto& brgemm_in0_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->input(0));
        const auto& brgemm_in1_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->input(1));
        const auto& brgemm_out_desc = PortDescriptorUtils::get_port_descriptor_ptr(brgemm->output(0));

        auto layout_a = brgemm_in0_desc->get_layout();
        auto layout_b = brgemm_in1_desc->get_layout();
        auto layout_c = brgemm_out_desc->get_layout();

        const auto etype_a = brgemm->get_input_element_type(0);
        const auto etype_b = brgemm->get_input_element_type(1);

        bool are_wei_constant = false;
        auto brgemm_parent_1 = brgemm->get_input_node_shared_ptr(1);
        // EnforcePrecision pass might insert ConvertSaturation before Brgemm
        const auto wei_convert = ov::as_type_ptr<ov::snippets::op::ConvertSaturation>(brgemm_parent_1);
        if (wei_convert) {
            brgemm_parent_1 = wei_convert->get_input_node_shared_ptr(0);
        }
        // Weights are repacked at compile time outside the Subgraph body (by EliminateBrgemmCopyB and
        // RepackMatMulWeights passes) only if BrgemmCopyB input is Parameter (optionally via RankNormalization)
        const bool is_wei_param_directly = [&]() {
            auto node = brgemm_parent_1;
            if (ov::is_type<ov::snippets::op::RankNormalization>(node)) {
                node = node->get_input_node_shared_ptr(0);
            }
            return ov::is_type<ov::op::v0::Parameter>(node);
        }();
        const auto shape_infer_leaf =
            ov::snippets::utils::get_leaf_node_of_first_parent_shape_infer_seq(brgemm_parent_1);
        if (shape_infer_leaf) {
            brgemm_parent_1 = shape_infer_leaf->input_value(0).get_node_shared_ptr();
        }
        if (const auto param = ov::as_type_ptr<ov::op::v0::Parameter>(brgemm_parent_1)) {
            const auto param_idx = static_cast<size_t>(model->get_parameter_index(param));
            OPENVINO_ASSERT(param_idx < model->get_parameters().size(),
                            "Parameter index is invalid in BrgemmToBrgemmCPU transformation");
            are_wei_constant = m_constant_inputs_idxs.count(param_idx) > 0;
        }
        // ConvertSaturation on weights can be fused into BrgemmCopyB if:
        //  - BrgemmCopyB kernel supports such weights conversion on the current platform, or
        //  - weights are constant and will be repacked (and converted) at compile time by OneDNN reorder,
        //    so BrgemmCopyB kernel is not used for them.
        // Otherwise, ConvertSaturation is kept as a separate op in the body.
        const bool fuse_wei_convert =
            wei_convert && ((are_wei_constant && is_wei_param_directly) ||
                            brgemm_utils::BrgemmConfig::is_wei_conversion_supported(
                                etype_a,
                                etype_b,
                                wei_convert->get_input_element_type(0)));
        const auto orig_wei_type = fuse_wei_convert ? wei_convert->get_input_element_type(0) : etype_b;
        if (wei_convert && !fuse_wei_convert) {
            // Weights go through the explicit conversion in the body, so they can't be considered as constant
            are_wei_constant = false;
        }

        const auto brgemm_config = brgemm_utils::BrgemmConfig(etype_a,
                                                              etype_b,
                                                              orig_wei_type,
                                                              are_wei_constant,
                                                              BrgemmCopyB::is_transposed(layout_b));

        auto offset_a = brgemm->get_offset_a();
        auto offset_b = brgemm->get_offset_b();
        auto offset_c = brgemm->get_offset_c();

        auto brgemm_in0 = brgemm->input_value(0);
        auto brgemm_in1 = brgemm->input_value(1);

        std::shared_ptr<BrgemmCPU> brgemm_cpu = nullptr;
        std::shared_ptr<BrgemmCopyB> brgemm_copy_b = nullptr;

        if (brgemm_config.with_wei_repacking()) {
            // BrgemmCopyB performs precision conversion, so we can skip explicit convert in this case
            if (fuse_wei_convert) {
                const auto* convert = ov::as_type<ov::snippets::op::ConvertSaturation>(brgemm_in1.get_node());
                OPENVINO_ASSERT(convert, "Expected ConvertSaturation on Brgemm weights input");
                brgemm_in1 = convert->input_value(0);
            }
            brgemm_copy_b =
                std::make_shared<BrgemmCopyB>(brgemm_in1, brgemm_config, layout_b, PortDescriptor{0, offset_b});
            PortDescriptorUtils::set_port_descriptor(brgemm_copy_b->input(0),
                                                     brgemm_in1_desc->get_subtensor(),
                                                     layout_b);
            for (const auto& out : brgemm_copy_b->outputs()) {
                snippets::utils::set_full_port_desc(out);
            }

            brgemm_in1 = brgemm_copy_b->output(0);
            layout_b.clear();
            offset_b = 0;
        }

        if (brgemm_config.is_amx()) {
            const auto scratch = std::make_shared<snippets::op::Buffer>(ov::Shape{BrgemmCPU::SCRATCH_BYTE_SIZE});
            brgemm_cpu = std::make_shared<BrgemmCPU>(ov::OutputVector{brgemm_in0, brgemm_in1, scratch},
                                                     brgemm_config,
                                                     std::vector<PortDescriptor>{{0, offset_a}, {0, offset_b}, {0, 0}},
                                                     PortDescriptor{0, offset_c},
                                                     layout_a,
                                                     layout_b,
                                                     layout_c);

            snippets::utils::set_full_port_desc(scratch->output(0));
            snippets::utils::set_full_port_desc(brgemm_cpu->input(2));
        } else if (brgemm_config.with_compensations()) {
            OPENVINO_ASSERT(brgemm_copy_b, "BrgemmCopyB is required");
            brgemm_cpu = std::make_shared<BrgemmCPU>(ov::OutputVector{brgemm_in0, brgemm_in1, brgemm_copy_b->output(1)},
                                                     brgemm_config,
                                                     std::vector<PortDescriptor>{{0, offset_a}, {0, offset_b}, {0, 0}},
                                                     PortDescriptor{0, offset_c},
                                                     layout_a,
                                                     layout_b,
                                                     layout_c);
        } else {
            brgemm_cpu = std::make_shared<BrgemmCPU>(ov::OutputVector{brgemm_in0, brgemm_in1},
                                                     brgemm_config,
                                                     std::vector<PortDescriptor>{{0, offset_a}, {0, offset_b}},
                                                     PortDescriptor{0, offset_c},
                                                     layout_a,
                                                     layout_b,
                                                     layout_c);
        }
        // need to run validate_and_infer_types manually: either input shapes were updated or
        // output Layout was updated (out shape will be updated in validate_and_infer_types())
        if (brgemm_copy_b) {
            snippets::utils::set_full_port_desc(brgemm_cpu->input(1));
            brgemm_copy_b->validate_and_infer_types();
        } else {
            PortDescriptorUtils::set_port_descriptor(brgemm_cpu->input(1), brgemm_in1_desc->get_subtensor(), layout_b);
        }
        PortDescriptorUtils::set_port_descriptor(brgemm_cpu->input(0), brgemm_in0_desc->get_subtensor(), layout_a);
        PortDescriptorUtils::set_port_descriptor(brgemm_cpu->output(0), brgemm_out_desc->get_subtensor(), layout_c);

        brgemm_cpu->validate_and_infer_types();
        brgemm_cpu->set_friendly_name(brgemm->get_friendly_name());
        ov::copy_runtime_info(brgemm, brgemm_cpu);
        ov::replace_node(brgemm, brgemm_cpu);

        status = true;
    }

    return status;
}
}  // namespace ov::intel_cpu
