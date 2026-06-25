// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/convert_gather_fusion.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace test {

std::string ConvertGatherFusionTest::getTestCaseName(const testing::TestParamInfo<ConvertGatherFusionParams>& obj) {
    const auto& [weights_shape, indices_shape, axis, weights_precision, output_precision, targetDevice] = obj.param;

    std::ostringstream result;
    result << "weights=" << ov::test::utils::vec2str(weights_shape) << "_";
    result << "indices=" << ov::test::utils::vec2str(indices_shape) << "_";
    result << "axis=" << axis << "_";
    result << "wPrec=" << weights_precision << "_";
    result << "outPrec=" << output_precision << "_";
    result << "device=" << targetDevice;
    return result.str();
}

void ConvertGatherFusionTest::SetUp() {
    const auto& [weights_shape, indices_shape, axis, weights_precision, output_precision, _targetDevice] = this->GetParam();
    targetDevice = _targetDevice;

    // Create model: Constant(bf16/f16) -> Convert(fp32) -> Gather
    ov::ParameterVector params;
    auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, indices_shape);
    params.push_back(indices);
    auto weights_const = ov::test::utils::create_and_fill_tensor(weights_precision, weights_shape);
    auto weights = std::make_shared<ov::op::v0::Constant>(weights_const);
    auto convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32);
    auto axis_const = ov::op::v0::Constant::create(ov::element::i32, {}, {axis});
    auto gather = std::make_shared<ov::op::v8::Gather>(convert, indices, axis_const);

    function = std::make_shared<ov::Model>(gather, params, "ConvertGatherFusion");
}

void ConvertGatherFusionTest::TearDown() {
    auto runtime_model = compiledModel.get_runtime_model();
    ASSERT_NE(nullptr, runtime_model);

    int convert_before_gather_count = 0;
    for (const auto& node : runtime_model->get_ops()) {
        if (node->get_type_info().name == std::string("Convert")) {
            auto convert_in_prec = node->get_input_element_type(0);
            auto convert_out_prec = node->get_output_element_type(0);
            if ((convert_in_prec == ov::element::bf16 || convert_in_prec == ov::element::f16) &&
                convert_out_prec == ov::element::f32) {
                for (const auto& output : node->outputs()) {
                    for (const auto& target_input : output.get_target_inputs()) {
                        auto child_node = target_input.get_node();
                        if (child_node->get_type_info().name == std::string("Gather")) {
                            convert_before_gather_count++;
                            break;
                        }
                    }
                }
            }
        }
    }

    EXPECT_EQ(0, convert_before_gather_count)
        << "Found " << convert_before_gather_count
        << " Convert(bf16/f16->fp32) node(s) before Gather. "
        << "FuseConvertAndGather fusion did not work!";
}

}  // namespace test
}  // namespace ov
