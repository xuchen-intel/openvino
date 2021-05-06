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
        CPUSpecificParams,
        std::map<std::string, std::string>> TopKLayerCPUTestParamsSet;

class TopKLayerCPUTest : public testing::WithParamInterface<TopKLayerCPUTestParamsSet>,
                                     virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<TopKLayerCPUTestParamsSet> obj) {
        LayerTestsDefinitions::TopKParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, cpuParams, additionalConfig) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::TopKLayerTest::getTestCaseName(
                     testing::TestParamInfo<LayerTestsDefinitions::TopKParams>(basicParamsSet, 0));

        result << CPUTestsBase::getTestCaseName(cpuParams);

        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto &item : additionalConfig) {
                if (item.second == PluginConfigParams::YES)
                    result << "_" << item.first << "=" << item.second;
            }
        }

        return result.str();
    }

protected:
    void SetUp() override {
        LayerTestsDefinitions::TopKParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, cpuParams, additionalConfig) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        InferenceEngine::SizeVector inputShape;
        InferenceEngine::Precision netPrecision;
        int64_t keepK, axis;
        ngraph::opset4::TopK::Mode mode;
        std::tie(keepK, axis, mode, sort, netPrecision, inPrc, outPrc, inLayout, inputShape, targetDevice) = basicParamsSet;

        if (additionalConfig[PluginConfigParams::KEY_ENFORCE_BF16] == PluginConfigParams::YES)
            inPrc = outPrc = netPrecision = Precision::BF16;
        else
            inPrc = outPrc = netPrecision;
        configuration.insert(additionalConfig.begin(), additionalConfig.end());

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

        selectedType += "_";
        selectedType += netPrecision.name();
    }

    void Validate() override {
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

    void Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expectedOutputs,
                 const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs) override {
        Compare(expectedOutputs, actualOutputs, threshold);
    }

    void Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expectedOutputs,
                                   const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs,
                                   float threshold) {
        for (std::size_t outputIndex = 0; outputIndex < expectedOutputs.size(); ++outputIndex) {
            const auto &expected = expectedOutputs[outputIndex];
            const auto &actual = actualOutputs[outputIndex];
            Compare(expected, actual, threshold);
        }
    }

    void Compare(const std::pair<ngraph::element::Type, std::vector<std::uint8_t>> &expected,
                                   const InferenceEngine::Blob::Ptr &actual,
                                   float threshold) {
        const auto &precision = actual->getTensorDesc().getPrecision();
        auto k =  static_cast<float>(expected.first.size()) / precision.size();
        // W/A for int4, uint4
        if (expected.first == ngraph::element::Type_t::u4 || expected.first == ngraph::element::Type_t::i4) {
            k /= 2;
        } else if (expected.first == ngraph::element::Type_t::undefined || expected.first == ngraph::element::Type_t::dynamic) {
            k = 1;
        }
        ASSERT_EQ(expected.second.size(), actual->byteSize() * k);

        auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
        IE_ASSERT(memory);
        const auto lockedMemory = memory->wmap();
        const auto actualBuffer = lockedMemory.as<const std::uint8_t *>();

        const auto &size = actual->size();
        switch (precision) {
            case InferenceEngine::Precision::FP32:
                callCompare<float>(expected, reinterpret_cast<const float *>(actualBuffer), size, threshold);
                break;
            case InferenceEngine::Precision::I32:
                callCompare<int32_t>(expected, reinterpret_cast<const int32_t *>(actualBuffer), size, threshold);
                break;
            case InferenceEngine::Precision::I64:
                callCompare<int64_t>(expected, reinterpret_cast<const int64_t *>(actualBuffer), size, threshold);
                break;
            case InferenceEngine::Precision::I8:
                callCompare<int8_t>(expected, reinterpret_cast<const int8_t *>(actualBuffer), size, threshold);
                break;
            case InferenceEngine::Precision::U16:
                callCompare<uint16_t>(expected, reinterpret_cast<const uint16_t *>(actualBuffer), size, threshold);
                break;
            case InferenceEngine::Precision::I16:
                callCompare<int16_t>(expected, reinterpret_cast<const int16_t *>(actualBuffer), size, threshold);
                break;
            case InferenceEngine::Precision::BOOL:
            case InferenceEngine::Precision::U8:
                callCompare<uint8_t>(expected, reinterpret_cast<const uint8_t *>(actualBuffer), size, threshold);
                break;
            case InferenceEngine::Precision::U64:
                callCompare<uint64_t>(expected, reinterpret_cast<const uint64_t *>(actualBuffer), size, threshold);
                break;
            case InferenceEngine::Precision::BF16:
                callCompare<ngraph::bfloat16>(expected, reinterpret_cast<const ngraph::bfloat16 *>(actualBuffer), size, threshold);
                break;
            case InferenceEngine::Precision::FP16:
                callCompare<ngraph::float16>(expected, reinterpret_cast<const ngraph::float16 *>(actualBuffer), size, threshold);
                break;
            default:
                FAIL() << "Comparator for " << precision << " precision isn't supported";
        }
    }

    template <typename T_IE>
    void callCompare(const std::pair<ngraph::element::Type, std::vector<std::uint8_t>> &expected,
                            const T_IE* actualBuffer, size_t size, float threshold) {
        auto expectedBuffer = expected.second.data();
        switch (expected.first) {
            case ngraph::element::Type_t::i64:
                Compare<T_IE, int64_t>(reinterpret_cast<const int64_t *>(expectedBuffer),
                                                         actualBuffer, size, threshold);
                break;
            case ngraph::element::Type_t::i32:
                Compare<T_IE, int32_t>(reinterpret_cast<const int32_t *>(expectedBuffer),
                                                         actualBuffer, size, threshold);
                break;
            case ngraph::element::Type_t::i16:
                Compare<T_IE, int16_t>(reinterpret_cast<const int16_t *>(expectedBuffer),
                                                         actualBuffer, size, threshold);
                break;
            case ngraph::element::Type_t::i8:
                Compare<T_IE, int8_t>(reinterpret_cast<const int8_t *>(expectedBuffer),
                                                        actualBuffer, size, threshold);
                break;
            case ngraph::element::Type_t::u64:
                Compare<T_IE, uint64_t>(reinterpret_cast<const uint64_t *>(expectedBuffer),
                                                          actualBuffer, size, threshold);
                break;
            case ngraph::element::Type_t::u32:
                Compare<T_IE, uint32_t>(reinterpret_cast<const uint32_t *>(expectedBuffer),
                                                          actualBuffer, size, threshold);
                break;
            case ngraph::element::Type_t::u16:
                Compare<T_IE, uint16_t>(reinterpret_cast<const uint16_t *>(expectedBuffer),
                                                          actualBuffer, size, threshold);
                break;
            case ngraph::element::Type_t::boolean:
            case ngraph::element::Type_t::u8:
                Compare<T_IE, uint8_t>(reinterpret_cast<const uint8_t *>(expectedBuffer),
                                                         actualBuffer, size, threshold);
                break;
            case ngraph::element::Type_t::f64:
                Compare<T_IE, double>(reinterpret_cast<const double *>(expectedBuffer),
                                                       actualBuffer, size, threshold);
                break;
            case ngraph::element::Type_t::f32:
                Compare<T_IE, float>(reinterpret_cast<const float *>(expectedBuffer),
                                                       actualBuffer, size, threshold);
                break;
            case ngraph::element::Type_t::f16:
                Compare<T_IE, ngraph::float16>(reinterpret_cast<const ngraph::float16 *>(expectedBuffer),
                                                                 actualBuffer, size, threshold);
                break;
            case ngraph::element::Type_t::bf16:
                Compare<T_IE, ngraph::bfloat16>(reinterpret_cast<const ngraph::bfloat16 *>(expectedBuffer),
                                                                  actualBuffer, size, threshold);
                break;
            case ngraph::element::Type_t::i4: {
                auto expectedOut = ngraph::helpers::convertOutputPrecision(
                        expected.second,
                        expected.first,
                        ngraph::element::Type_t::i8,
                        size);
                Compare<T_IE, int8_t>(reinterpret_cast<const int8_t *>(expectedOut.data()),
                                                        actualBuffer, size, threshold);
                break;
            }
            case ngraph::element::Type_t::u4: {
                auto expectedOut = ngraph::helpers::convertOutputPrecision(
                        expected.second,
                        expected.first,
                        ngraph::element::Type_t::u8,
                        size);
                Compare<T_IE, uint8_t>(reinterpret_cast<const uint8_t *>(expectedOut.data()),
                                                         actualBuffer, size, threshold);
                break;
            }
            case ngraph::element::Type_t::dynamic:
            case ngraph::element::Type_t::undefined:
                Compare<T_IE, T_IE>(reinterpret_cast<const T_IE *>(expectedBuffer), actualBuffer, size, threshold);
                break;
            default: FAIL() << "Comparator for " << expected.first << " precision isn't supported";
        }
        return;
    }

    template<class T_IE, class T_NGRAPH>
    void Compare(const T_NGRAPH *expected, const T_IE *actual, std::size_t size, float threshold) {
        for (size_t o = 0; o < outer_size; o++) {
            for (size_t i = 0; i < inner_size; i++) {
                std::vector<T_NGRAPH> v_expected;
                std::vector<T_IE> v_actual;
                for (size_t k = 0; k < top_k; k++) {
                    v_expected.push_back(expected[(o * top_k + k) * inner_size + i]);
                    v_actual.push_back(actual[(o * top_k + k) * inner_size + i]);
                }
                std::sort(v_expected.begin(), v_expected.end());
                std::sort(v_actual.begin(), v_actual.end());
                for (size_t k = 0; k < top_k; k++) {
                    const T_NGRAPH &ref = v_expected[k];
                    const auto &res = v_actual[k];
                    const auto absoluteDifference = CommonTestUtils::ie_abs(res - ref);
                    if (absoluteDifference <= threshold) {
                        continue;
                    }
                    double max;
                    if (sizeof(T_IE) < sizeof(T_NGRAPH)) {
                        max = std::max(CommonTestUtils::ie_abs(T_NGRAPH(res)), CommonTestUtils::ie_abs(ref));
                    } else {
                        max = std::max(CommonTestUtils::ie_abs(res), CommonTestUtils::ie_abs(T_IE(ref)));
                    }
                    double diff = static_cast<float>(absoluteDifference) / max;
                    if (max == 0 || (diff > static_cast<float>(threshold)) ||
                        (std::isnan(static_cast<float>(res)) ^ std::isnan(static_cast<float>(ref)))) {
                        IE_THROW() << "Relative comparison of values expected: " << std::to_string(ref) << " and actual: " << std::to_string(res)
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

std::vector<CPUSpecificParams> filterCPUInfoForDevice() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x}, {nchw, nchw}, {"jit_avx512"}, "jit_avx512"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x}, {nhwc, nhwc}, {"jit_avx512"}, "jit_avx512"});
        resCPUParams.push_back(CPUSpecificParams{{nChw16c, x}, {nChw16c, nChw16c}, {"jit_avx512"}, "jit_avx512"});
    } else if (with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x}, {nchw, nchw}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x}, {nhwc, nhwc}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, x}, {nChw8c, nChw8c}, {"jit_avx2"}, "jit_avx2"});
    } else if (with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x}, {nchw, nchw}, {"jit_sse42"}, "jit_sse42"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x}, {nhwc, nhwc}, {"jit_sse42"}, "jit_sse42"});
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, x}, {nChw8c, nChw8c}, {"jit_sse42"}, "jit_sse42"});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x}, {nchw, nchw}, {"ref"}, "ref"});
    }
    return resCPUParams;
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::BF16
};

std::vector<std::map<std::string, std::string>> additionalConfig = {
    {{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::NO}},
    {{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES}}
};

const std::vector<int64_t> axes = {0, 1, 2, 3};
const std::vector<int64_t> k = {1, 5, 7, 18, 21};

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
            ::testing::Values(std::vector<size_t>({21, 21, 21, 21})),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice()),
        ::testing::ValuesIn(additionalConfig)),
    TopKLayerCPUTest::getTestCaseName);

} // namespace

} // namespace CPULayerTestsDefinitions
