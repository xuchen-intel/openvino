// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include <map>

#include "gtest/gtest.h"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {
using ov::test::utils::ComparisonTypes;

static std::map<ComparisonTypes, std::string> comparisonNames = {
        {ComparisonTypes::EQUAL,         "EQUAL"},
        {ComparisonTypes::NOT_EQUAL,     "NOT_EQUAL"},
        {ComparisonTypes::IS_FINITE,     "IS_FINITE"},
        {ComparisonTypes::IS_INF,        "IS_INF"},
        {ComparisonTypes::IS_NAN,        "IS_NAN"},
        {ComparisonTypes::LESS,          "LESS"},
        {ComparisonTypes::LESS_EQUAL,    "LESS_EQUAL"},
        {ComparisonTypes::GREATER,       "GREATER"},
        {ComparisonTypes::GREATER_EQUAL, "GREATER_EQUAL"},
};

typedef std::tuple<
    std::vector<InputShape>,             // Input shapes tuple
    ov::test::utils::ComparisonTypes,    // Comparison op type
    ov::test::utils::InputLayerType,     // Second input type
    ov::element::Type,                   // Model type
    std::string,                         // Device name
    std::map<std::string, std::string>   // Additional network configuration
> ComparisonTestParams;

class ComparisonLayerTest : public testing::WithParamInterface<ComparisonTestParams>,
    virtual public ov::test::SubgraphBaseTest {
protected:
    void SetUp() override;
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ComparisonTestParams> &obj);
};
} // namespace test
} // namespace ov
