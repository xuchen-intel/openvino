// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/fake_convert_decomposition.hpp"

#include "common_test_utils/ov_test_utils.hpp"

using namespace ov;

using FakeConvertDecompositionParamsSet = std::tuple<element::Type_t,
                                                     Shape>;

class FakeConvertDecompositionTest : public ov::test::TestsCommon,
                                     public ::testing::WithParamInterface<FakeConvertDecompositionParamsSet> {
public:
    static std::string getTestCaseName(::testing::TestParamInfo<FakeConvertDecompositionParamsSet> obj) {
        std::ostringstream result;
        return result.str();
    }

protected:
    void SetUp() override {
    }
};

TEST_P(FakeConvertDecompositionTest, CompareFunctions) {}
