// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/convert_gather_fusion.hpp"

#include <vector>

using namespace ov::test;

namespace {

const std::vector<ov::Shape> weights_shapes = {
    {1024, 512},
};

const std::vector<ov::Shape> indices_shapes = {
    {256, 1},
};

const std::vector<int> axes = {0};

const std::vector<ov::element::Type> weights_precisions = {
    ov::element::bf16,
    ov::element::f16
};

const std::vector<ov::element::Type> output_precisions = {
    ov::element::f32
};

INSTANTIATE_TEST_SUITE_P(
    smoke_ConvertGatherFusion,
    ConvertGatherFusionTest,
    ::testing::Combine(
        ::testing::ValuesIn(weights_shapes),
        ::testing::ValuesIn(indices_shapes),
        ::testing::ValuesIn(axes),
        ::testing::ValuesIn(weights_precisions),
        ::testing::ValuesIn(output_precisions),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
    ),
    ConvertGatherFusionTest::getTestCaseName);

}  // namespace
