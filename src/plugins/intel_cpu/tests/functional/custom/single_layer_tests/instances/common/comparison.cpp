// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/comparison.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace comparison {

const auto basicCasesSnippets = ::testing::Combine(
    ::testing::ValuesIn(static_shapes_to_test_representation(inputShape())),
    ::testing::ValuesIn(comparisonTypes()),
    ::testing::ValuesIn(netPrecisions()),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams4D())),
    ::testing::Values(true)
);

INSTANTIATE_TEST_SUITE_P(smoke_Comparison_Snippets_CPU, ComparisonLayerCPUTest, basicCasesSnippets, ComparisonLayerCPUTest::getTestCaseName);

}  // namespace comparison
}  // namespace test
}  // namespace ov

// }  // namespace
