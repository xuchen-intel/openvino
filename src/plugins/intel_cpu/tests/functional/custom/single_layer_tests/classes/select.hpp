// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "gtest/gtest.h"

namespace ov {
namespace test {

using SelectLayerCPUTestParamSet =
    std::tuple<std::vector<InputShape>,           // Input shapes
               ov::element::Type,                 // Net precision
               ov::element::Type,                 // Input precision
               ov::element::Type,                 // Output precision
               ov::op::AutoBroadcastSpec,         // Broadcast
               CPUTestUtils::CPUSpecificParams,
               bool>;

class SelectLayerCPUTest : public testing::WithParamInterface<SelectLayerCPUTestParamSet>,
                           virtual public ov::test::SubgraphBaseTest,
                           public CPUTestUtils::CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SelectLayerCPUTestParamSet> &obj);

protected:
    void SetUp() override;

private:
    std::string getPrimitiveType();
};

namespace select {

const std::vector<std::vector<ov::Shape>>& inputShape();

const std::vector<ov::element::Type>& netPrecisions();

const std::vector<ov::op::AutoBroadcastSpec>& broadcast();

const std::vector<CPUTestUtils::CPUSpecificParams>& cpuParams4D();

}  // namespace select
}  // namespace test
}  // namespace ov
