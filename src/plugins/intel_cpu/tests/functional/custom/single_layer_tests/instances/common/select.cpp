// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/select.hpp"

namespace ov {
namespace test {
namespace select {

const auto basicCasesSnippets = ::testing::Combine(
    ::testing::ValuesIn(static_shapes_to_test_representation(inputShape())),
    ::testing::ValuesIn(netPrecisions()),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(broadcast()),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams4D())),
    ::testing::Values(true)
);

INSTANTIATE_TEST_SUITE_P(smoke_Select_Snippets_CPU, SelectLayerCPUTest, basicCasesSnippets, SelectLayerCPUTest::getTestCaseName);

}  // namespace select
}  // namespace test
}  // namespace ov
