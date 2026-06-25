// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/convert_gather_fusion.hpp"

namespace ov {
namespace test {

TEST_P(ConvertGatherFusionTest, CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
