// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>

#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace {

class CompatibilityStringCPU : public ::testing::Test {
public:
    std::shared_ptr<ov::Model> model;

    void SetUp() override {
        model = ov::test::utils::make_conv_pool_relu();
    }
};

TEST_F(CompatibilityStringCPU, RuntimeRequirementsIsSupportedAndNonEmpty) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::Core core;
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_CPU));

    auto supported = compiled_model.get_property(ov::supported_properties);
// std::cout << "supported.size(): " << supported.size() << std::endl;
// for (auto && prop : supported) {
//     std::cout << "supported property: " << prop << std::endl;
// }
// std::cout << "ov::runtime_requirements.name(): " << ov::runtime_requirements.name() << std::endl;
    ASSERT_NE(std::find(supported.begin(), supported.end(), ov::runtime_requirements.name()), supported.end());

    std::string requirements;
    OV_ASSERT_NO_THROW(requirements = compiled_model.get_property(ov::runtime_requirements));
std::cout << "requirements: " << requirements << std::endl;
    ASSERT_FALSE(requirements.empty());
    std::cout << "[ INFO     ] CPU ov::runtime_requirements = " << requirements << std::endl;
}

}  // namespace
