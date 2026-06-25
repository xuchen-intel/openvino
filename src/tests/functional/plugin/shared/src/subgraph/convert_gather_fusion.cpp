// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/convert_gather_fusion.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "transformations/rt_info/decompression.hpp"

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

    // Create model mimicking LLM weight tying (e.g., bitnet):
    //   Constant(bf16/f16)
    //         |
    //   Convert(fp32) ──┬─> Gather (embedding)
    //                   └─> MatMul (lm_head)
    // The shared Convert with multiple consumers prevents MoveDecompressionAfterGather
    // (which requires consumers_count(1)), so the Convert stays in front of Gather and
    // must be removed by FuseConvertAndGather instead.
    ov::ParameterVector params;
    auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, indices_shape);
    params.push_back(indices);
    // Second consumer for the Convert — MatMul on a Parameter (lm_head-like).
    auto matmul_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, weights_shape[1]});
    params.push_back(matmul_input);

    auto weights_const = ov::test::utils::create_and_fill_tensor(weights_precision, weights_shape);
    auto weights = std::make_shared<ov::op::v0::Constant>(weights_const);
    auto convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32);
    // Mark as decompression to prevent ConstantFolding from folding
    // Constant(bf16) + Convert(f32) into a single Constant(f32). This mimics
    // how CompressFloatConstants marks decompression Converts in real models.
    ov::mark_as_decompression(convert);

    auto axis_const = ov::op::v0::Constant::create(ov::element::i32, {}, {axis});
    auto gather = std::make_shared<ov::op::v8::Gather>(convert, indices, axis_const);

    auto matmul = std::make_shared<ov::op::v0::MatMul>(matmul_input, convert, false, true);

    auto gather_result = std::make_shared<ov::op::v0::Result>(gather);
    auto matmul_result = std::make_shared<ov::op::v0::Result>(matmul);
    function = std::make_shared<ov::Model>(ov::ResultVector{gather_result, matmul_result},
                                           params,
                                           "ConvertGatherFusion");
}

void ConvertGatherFusionTest::TearDown() {
    const auto model = compiledModel.get_runtime_model();
    for (const auto& node : model->get_ordered_ops()) {
        const auto& rt_info = node->get_rt_info();
        const auto layer_type = rt_info.find("layerType")->second.as<std::string>();
        EXPECT_NE(layer_type, "Convert");
    }
}

}  // namespace test
}  // namespace ov
