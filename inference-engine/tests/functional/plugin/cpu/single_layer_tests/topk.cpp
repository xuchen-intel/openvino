// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/topk.hpp>
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        LayerTestsDefinitions::TopKParams,
        CPUSpecificParams> TopKLayerCPUTestParamsSet;

class TopKLayerCPUTest : public testing::WithParamInterface<TopKLayerCPUTestParamsSet>,
                                     virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<TopKLayerCPUTestParamsSet> obj) {
        LayerTestsDefinitions::TopKParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::TopKLayerTest::getTestCaseName(
                     testing::TestParamInfo<LayerTestsDefinitions::TopKParams>(basicParamsSet, 0));

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() {
        LayerTestsDefinitions::TopKParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        InferenceEngine::SizeVector inputShape;
        InferenceEngine::Precision netPrecision;
        int64_t keepK, axis;
        ngraph::opset4::TopK::Mode mode;
        std::tie(keepK, axis, mode, sort, netPrecision, inPrc, outPrc, inLayout, inputShape, targetDevice) = basicParamsSet;

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
        topk->get_rt_info() = getCPUInfo();

        ngraph::ResultVector results;
        for (int i = 0; i < topk->get_output_size(); i++) {
            results.push_back(std::make_shared<ngraph::opset4::Result>(topk->output(i)));
        }
        function = std::make_shared<ngraph::Function>(results, params, "TopK");
    }

    void Validate() {
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
        if (sort == ngraph::opset4::TopK::SortType::SORT_VALUES) {
            LayerTestsCommon::Compare(expectedOutputs[0], actualOutputs[0]);
        } else {
            Compare(expectedOutputs[0], actualOutputs[0]);
        }
    }

    void Compare(const std::vector<std::uint8_t> &expected, const InferenceEngine::Blob::Ptr &actual) {
        Compare(expected, actual, threshold);
    }

    void Compare(const std::vector<std::uint8_t> &expected,
                                   const InferenceEngine::Blob::Ptr &actual,
                                   float threshold) {
        ASSERT_EQ(expected.size(), actual->byteSize());
        const auto &expectedBuffer = expected.data();

        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
        IE_ASSERT(memory);
        const auto lockedMemory = memory->wmap();
        const auto actualBuffer = lockedMemory.as<const std::uint8_t *>();

        const auto &precision = actual->getTensorDesc().getPrecision();
        const auto &size = actual->size();
        switch (precision) {
            case InferenceEngine::Precision::FP32:
                Compare<float>(reinterpret_cast<const float *>(expectedBuffer),
                               reinterpret_cast<const float *>(actualBuffer), size, threshold);
                break;
            case InferenceEngine::Precision::I32:
                Compare<int32_t>(reinterpret_cast<const int32_t *>(expectedBuffer),
                                 reinterpret_cast<const int32_t *>(actualBuffer), size, 0);
                break;
            case InferenceEngine::Precision::I64:
                Compare<int64_t>(reinterpret_cast<const int64_t *>(expectedBuffer),
                                 reinterpret_cast<const int64_t *>(actualBuffer), size, 0);
                break;
            case InferenceEngine::Precision::I8:
                Compare<int8_t>(reinterpret_cast<const int8_t *>(expectedBuffer),
                                reinterpret_cast<const int8_t *>(actualBuffer), size, 0);
                break;
            case InferenceEngine::Precision::U16:
                Compare<uint16_t>(reinterpret_cast<const uint16_t *>(expectedBuffer),
                                  reinterpret_cast<const uint16_t *>(actualBuffer), size, 0);
                break;
            case InferenceEngine::Precision::I16:
                Compare<int16_t>(reinterpret_cast<const int16_t *>(expectedBuffer),
                                 reinterpret_cast<const int16_t *>(actualBuffer), size, 0);
                break;
            case InferenceEngine::Precision::BOOL:
            case InferenceEngine::Precision::U8:
                Compare<uint8_t>(reinterpret_cast<const uint8_t *>(expectedBuffer),
                                 reinterpret_cast<const uint8_t *>(actualBuffer), size, 0);
                break;
            case InferenceEngine::Precision::U64:
                Compare<uint64_t>(reinterpret_cast<const uint64_t *>(expectedBuffer),
                                  reinterpret_cast<const uint64_t *>(actualBuffer), size, 0);
                break;
            case InferenceEngine::Precision::BF16:
                Compare(reinterpret_cast<const ngraph::bfloat16 *>(expectedBuffer),
                        reinterpret_cast<const ngraph::bfloat16 *>(actualBuffer), size, ngraph::bfloat16(threshold));
                break;
            case InferenceEngine::Precision::FP16:
                Compare(reinterpret_cast<const ngraph::float16 *>(expectedBuffer),
                        reinterpret_cast<const ngraph::float16 *>(actualBuffer), size, ngraph::float16(threshold));
                break;
            default:
                FAIL() << "Comparator for " << precision << " precision isn't supported";
        }
    }

    template<class T>
    void Compare(const T *expected, const T *actual, std::size_t size, T threshold) {
        for (size_t o = 0; o < outer_size; o++) {
            for (size_t i = 0; i < inner_size; i++) {
                std::vector<T> v_expected;
                std::vector<T> v_actual;
                for (size_t k = 0; k < top_k; k++) {
                    v_expected.push_back(expected[(o * top_k + k) * inner_size + i]);
                    v_actual.push_back(actual[(o * top_k + k) * inner_size + i]);
                }
                std::sort(v_expected.begin(), v_expected.end());
                std::sort(v_actual.begin(), v_actual.end());
                for (size_t k = 0; k < top_k; k++) {
                    const auto &ref = v_expected[k];
                    const auto &res = v_actual[k];
                    const auto absoluteDifference = CommonTestUtils::ie_abs(res - ref);
                    if (absoluteDifference <= threshold) {
                        continue;
                    }

                    const auto max = std::max(CommonTestUtils::ie_abs(res), CommonTestUtils::ie_abs(ref));
                    float diff = static_cast<float>(absoluteDifference) / static_cast<float>(max);
                    if (max == 0 || (diff > static_cast<float>(threshold))) {
                        IE_THROW() << "Relative comparison of values expected: " << ref << " and actual: " << res
                                           << " at index " << k << " with threshold " << threshold
                                           << " failed";
                    }
                }
            }
        }
    }

private:
    size_t axis_idx;
    size_t top_k;
    size_t outer_size = 1;
    size_t inner_size = 1;
    ngraph::opset4::TopK::SortType sort;
};

TEST_P(TopKLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "TopK");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> filterCPUInfoForDevice() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x}, {nchw, nchw}, {"jit_avx512"}, "jit_avx512_FP32"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x}, {nhwc, nhwc}, {"jit_avx512"}, "jit_avx512_FP32"});
        resCPUParams.push_back(CPUSpecificParams{{nChw16c, x}, {nChw16c, nChw16c}, {"jit_avx512"}, "jit_avx512_FP32"});
    } else if (with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x}, {nchw, nchw}, {"jit_avx2"}, "jit_avx2_FP32"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x}, {nhwc, nhwc}, {"jit_avx2"}, "jit_avx2_FP32"});
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, x}, {nChw8c, nChw8c}, {"jit_avx2"}, "jit_avx2_FP32"});
    } else if (with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x}, {nchw, nchw}, {"jit_sse42"}, "jit_sse42_FP32"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x}, {nhwc, nhwc}, {"jit_sse42"}, "jit_sse42_FP32"});
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, x}, {nChw8c, nChw8c}, {"jit_sse42"}, "jit_sse42_FP32"});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x}, {nchw, nchw}, {"ref"}, "ref_FP32"});
    }
    return resCPUParams;
}
/* ========== */

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32
};

const std::vector<int64_t> axes = {0, 1, 2, 3};
const std::vector<int64_t> k = {1, 5, 9, 10};

const std::vector<ngraph::opset4::TopK::Mode> modes = {
    ngraph::opset4::TopK::Mode::MIN,
    ngraph::opset4::TopK::Mode::MAX
};

const std::vector<ngraph::opset4::TopK::SortType> sortTypes = {
    ngraph::opset4::TopK::SortType::SORT_VALUES,
    ngraph::opset4::TopK::SortType::SORT_INDICES,
    ngraph::opset4::TopK::SortType::NONE
};

INSTANTIATE_TEST_CASE_P(smoke_TopK, TopKLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            ::testing::ValuesIn(k),
            ::testing::ValuesIn(axes),
            ::testing::ValuesIn(modes),
            ::testing::ValuesIn(sortTypes),
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(std::vector<size_t>({10, 10, 10, 10})),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice())),
    TopKLayerCPUTest::getTestCaseName);

} // namespace

} // namespace CPULayerTestsDefinitions
