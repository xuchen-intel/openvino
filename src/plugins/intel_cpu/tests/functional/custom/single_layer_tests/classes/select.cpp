// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "select.hpp"
#include "internal_properties.hpp"
#include "shared_test_classes/single_op/select.hpp"

namespace ov {
namespace test {

using namespace CPUTestUtils;

std::string SelectLayerCPUTest::getTestCaseName(const testing::TestParamInfo<SelectLayerCPUTestParamSet> &obj) {
    std::vector<ov::test::InputShape> inputShapes;
    ov::element::Type netPrecision, inPrecision, outPrecision;
    ov::op::AutoBroadcastSpec broadcast;
    CPUTestUtils::CPUSpecificParams cpuParams;
    bool enforceSnippets;
    std::tie(inputShapes, netPrecision, inPrecision, outPrecision, broadcast, cpuParams, enforceSnippets) = obj.param;
    std::ostringstream result;
    if (inputShapes.front().first.size() != 0) {
        result << "IS=(";
        for (const auto &shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result.seekp(-1, result.cur);
        result << ")_";
    }
    result << "TS=";
    for (const auto& shape : inputShapes) {
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
    }
    result << "netPRC=" << netPrecision.to_string() << "_";
    result << "inPRC=" << inPrecision.to_string() << "_";
    result << "outPRC=" << outPrecision.to_string() << "_";
    result << "Broadcast=" << broadcast.m_type;
    result << CPUTestUtils::CPUTestsBase::getTestCaseName(cpuParams);
    result << "_enforceSnippets=" << enforceSnippets;

    return result.str();
}

void SelectLayerCPUTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;

    std::vector<ov::test::InputShape> inputShapes;
    ov::element::Type netPrecision, inPrecision, outPrecision;
    ov::op::AutoBroadcastSpec broadcast;
    CPUTestUtils::CPUSpecificParams cpuParams;
    bool enforceSnippets;
    std::tie(inputShapes, netPrecision, inPrecision, outPrecision, broadcast, cpuParams, enforceSnippets) = this->GetParam();
    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

    inType  = inPrecision;
    outType = outPrecision;
    const auto primitiveType = getPrimitiveType();
    selectedType = primitiveType.empty() ? "" : primitiveType + "_" + netPrecision.to_string();

    init_input_shapes(inputShapes);

    if (enforceSnippets) {
        configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::IGNORE_CALLBACK));
    } else {
        configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::DISABLE));
    }

    ov::element::TypeVector types{ov::element::boolean, netPrecision, netPrecision};
    ov::ParameterVector parameters;
    for (size_t i = 0; i < types.size(); i++) {
        auto param_node = std::make_shared<ov::op::v0::Parameter>(types[i], inputDynamicShapes[i]);
        parameters.push_back(param_node);
    }
    auto select = std::make_shared<ov::op::v1::Select>(parameters[0], parameters[1], parameters[2], broadcast);
    select->get_rt_info() = getCPUInfo();
    function = makeNgraphFunction(netPrecision, parameters, select, "Select");
}

std::string SelectLayerCPUTest::getPrimitiveType() {
    return "jit";
}

TEST_P(SelectLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Eltwise");
}

namespace select {

const std::vector<std::vector<ov::Shape>>& inputShape() {
    static const std::vector<std::vector<ov::Shape>> shape {
        {{2, 4, 4, 1}, {2, 4, 4, 1}, {2, 4, 4, 1}},
        {{2, 17, 5, 4}, {2, 17, 5, 4}, {2, 17, 5, 4}},
    };

    return shape;
}

const std::vector<ov::element::Type>& netPrecisions() {
    static const std::vector<ov::element::Type> netPrecisions {
        ov::element::f32
    };

    return netPrecisions;
}

const std::vector<ov::op::AutoBroadcastSpec>& broadcast() {
    static const std::vector<ov::op::AutoBroadcastSpec> broadcast {
        ov::op::AutoBroadcastType::NONE
    };

    return broadcast;
}

const std::vector<CPUSpecificParams>& cpuParams4D() {
    static const std::vector<CPUSpecificParams> cpuParams4D {
        CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
        CPUSpecificParams({nchw}, {nchw}, {}, {})
    };

    return cpuParams4D;
}

}  // namespace select
}  // namespace test
}  // namespace ov
