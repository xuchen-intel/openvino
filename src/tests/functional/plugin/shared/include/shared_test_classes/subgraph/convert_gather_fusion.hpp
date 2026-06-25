// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

/*
 * Test Convert(bf16/f16 -> fp32) + Gather fusion
 *
 * Before fusion:
 *   Weights(BF16/F16) -> Convert(FP32) -> Gather(FP32 in, FP32 out)
 *
 * After fusion (via FuseConvertAndGather pass):
 *   Weights(BF16/F16) -> Gather(BF16/F16 in, FP32 out)
 */

using ConvertGatherFusionParams = std::tuple<
    ov::Shape,              // weights shape
    ov::Shape,              // indices shape
    int,                    // axis
    ov::element::Type,      // weights precision (bf16/f16)
    ov::element::Type,      // output precision (fp32)
    std::string             // device name
>;

class ConvertGatherFusionTest : public testing::WithParamInterface<ConvertGatherFusionParams>,
                                virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvertGatherFusionParams>& obj);

protected:
    void SetUp() override;
    void TearDown() override;
};

}  // namespace test
}  // namespace ov
