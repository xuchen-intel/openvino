// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include <map>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_enums.hpp"
#include "utils/cpu_test_utils.hpp"
#include "gtest/gtest.h"
#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {

using ComparisonLayerCPUTestParamSet =
    std::tuple<std::vector<InputShape>,          // Input shapes
               utils::ComparisonTypes,           // Comparison type
               ov::element::Type,                // Net precision
               ov::element::Type,                // Input precision
               ov::element::Type,                // Output precision
               CPUTestUtils::CPUSpecificParams,
               bool>;

class ComparisonLayerCPUTest : public testing::WithParamInterface<ComparisonLayerCPUTestParamSet>,
                               virtual public ov::test::SubgraphBaseTest,
                               public CPUTestUtils::CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ComparisonLayerCPUTestParamSet> &obj);

protected:
    void SetUp() override;

private:
    ov::element::Type netPrecision = ov::element::undefined;

    std::string getPrimitiveType();
};

namespace comparison {

const std::vector<std::vector<ov::Shape>>& inputShape();

const std::vector<utils::ComparisonTypes>& comparisonTypes();

const std::vector<ov::element::Type>& netPrecisions();

const std::vector<CPUTestUtils::CPUSpecificParams>& cpuParams4D();

}  // namespace comparison
}  // namespace test
}  // namespace ov
