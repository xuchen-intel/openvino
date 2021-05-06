// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {
typedef std::tuple<
        int64_t,                        // keepK
        int64_t,                        // axis
        ngraph::opset4::TopK::Mode,     // mode
        ngraph::opset4::TopK::SortType, // sort
        InferenceEngine::Precision,     // Net precision
        InferenceEngine::Precision,     // Input precision
        InferenceEngine::Precision,     // Output precision
        InferenceEngine::Layout,        // Input layout
        InferenceEngine::SizeVector,    // inputShape
        std::string                     // Target device name
> TopKParams;

class TopKLayerTest : public testing::WithParamInterface<TopKParams>,
                      virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<TopKParams> obj);

protected:
    void SetUp() override;
    void Validate() override;
    void Compare(const std::vector<std::uint8_t> &expected, const InferenceEngine::Blob::Ptr &actual) override;
    void Compare(const std::vector<std::uint8_t> &expected, const InferenceEngine::Blob::Ptr &actual, float threshold);

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

}  // namespace LayerTestsDefinitions
