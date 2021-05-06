// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/topk.hpp"

namespace LayerTestsDefinitions {
    std::string TopKLayerTest::getTestCaseName(const testing::TestParamInfo<TopKParams>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout;
    InferenceEngine::SizeVector inputShape;
    std::string targetDevice;
    int64_t keepK, axis;
    ngraph::opset4::TopK::Mode mode;
    ngraph::opset4::TopK::SortType sort;
    std::tie(keepK, axis, mode, sort, netPrecision, inPrc, outPrc, inLayout, inputShape, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "k=" << keepK << "_";
    result << "axis=" << axis << "_";
    result << "mode=" << mode << "_";
    result << "sort=" << sort << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void TopKLayerTest::SetUp() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    int64_t keepK, axis;
    ngraph::opset4::TopK::Mode mode;
    std::tie(keepK, axis, mode, sort, netPrecision, inPrc, outPrc, inLayout, inputShape, targetDevice) = this->GetParam();

    axis_idx = axis < 0 ? static_cast<size_t>(axis + static_cast<int64_t>(inputShape.size())) : static_cast<size_t>(axis);
    top_k = static_cast<size_t>(keepK);
    for (size_t i = 0; i < axis_idx; i++)
        outer_size *= inputShape[i];
    for (size_t i = axis_idx + 1; i < static_cast<size_t>(inputShape.size()); i++)
        inner_size *= inputShape[i];

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramIn = ngraph::helpers::convert2OutputVector(
                        ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    auto k = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{}, &keepK);
    auto topk = std::dynamic_pointer_cast<ngraph::opset4::TopK>(
            std::make_shared<ngraph::opset4::TopK>(paramIn[0], k, axis, mode, sort));

    ngraph::ResultVector results;
    for (int i = 0; i < topk->get_output_size(); i++) {
        results.push_back(std::make_shared<ngraph::opset4::Result>(topk->output(i)));
    }
    function = std::make_shared<ngraph::Function>(results, params, "TopK");
}

void TopKLayerTest::Validate() {
    auto expectedOutputs = CalculateRefs();
    const auto &actualOutputs = GetOutputs();

    if (expectedOutputs.empty()) {
        return;
    }

    IE_ASSERT(actualOutputs.size() == expectedOutputs.size())
    << "nGraph interpreter has " << expectedOutputs.size() << " outputs, while IE " << actualOutputs.size();

    // Spec TopK_3.md allows to use unstable sorting, thus
    // a. Skip comparing of index results, because an element in actual index tensor can be different with
    //    its counterpart in expected index tensor
    // b. If SortType is SORT_INDICES or NONE, the test program still needs to apply std::sort for all pairs
    //    of 1xk value vectors in expected and actual output tensor before comparing them
    std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> expectedOutput = {expectedOutputs[0]};
    std::vector<InferenceEngine::Blob::Ptr> actualOutput = {actualOutputs[0]};
    if (sort == ngraph::opset4::TopK::SortType::SORT_VALUES) {
        LayerTestsCommon::Compare(expectedOutput, actualOutput);
    } else {
        Compare(expectedOutput, actualOutput);
    }
}

void TopKLayerTest::Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expectedOutputs,
             const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs) {
    Compare(expectedOutputs, actualOutputs, threshold);
}

void TopKLayerTest::Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expectedOutputs,
                               const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs,
                               float threshold) {
    for (std::size_t outputIndex = 0; outputIndex < expectedOutputs.size(); ++outputIndex) {
        const auto &expected = expectedOutputs[outputIndex];
        const auto &actual = actualOutputs[outputIndex];
        Compare(expected, actual, threshold);
    }
}
}  // namespace LayerTestsDefinitions
