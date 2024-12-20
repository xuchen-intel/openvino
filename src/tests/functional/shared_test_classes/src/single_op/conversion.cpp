// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/conversion.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"

namespace ov {
namespace test {
namespace {
std::map<ov::test::utils::ConversionTypes, std::string> conversionNames = {
    {ov::test::utils::ConversionTypes::CONVERT, "Convert"},
    {ov::test::utils::ConversionTypes::CONVERT_LIKE, "ConvertLike"}};
}

static std::string special_value_to_string(const ov::test::SpecialValue& value) {
    if (value == SpecialValue::none) {
        return "none";
    } else if (value == SpecialValue::nan) {
        return "nan";
    } else if (value == SpecialValue::inf) {
        return "inf";
    } else if (value == SpecialValue::overflow) {
        return "overflow";
    }
    return "unknown";
}

template <typename T>
static T set_special_value(T& value, const ov::test::SpecialValue& special_value) {
    if (special_value == ov::test::SpecialValue::nan) {
        value = NAN;
    } else if (special_value == ov::test::SpecialValue::inf) {
        value = INFINITY;
    } else if (special_value == ov::test::SpecialValue::overflow) {
        value = value + std::numeric_limits<ov::float8_e5m2>::max();
    }
    return value;
}

template <typename T>
static void modify_value(ov::Tensor& tensor, const ov::test::SpecialValue& special_value) {
    T* dataPtr = static_cast<T*>(tensor.data());
    for (size_t i = 0; i < tensor.get_size(); i++) {
        set_special_value<T>(dataPtr[i], special_value);
    }
}

std::string ConversionLayerTest::getTestCaseName(const testing::TestParamInfo<ConversionParamsTuple>& obj) {
    ov::test::utils::ConversionTypes conversion_type;
    ov::element::Type input_type, convert_type;
    std::string device_name;
    std::vector<InputShape> shapes;
    ov::test::SpecialValue special_value;
    std::tie(conversion_type, shapes, input_type, convert_type, special_value, device_name) = obj.param;
    std::ostringstream result;
    result << "conversionOpType=" << conversionNames[conversion_type] << "_";
    result << "IS=(";
    for (size_t i = 0lu; i < shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({shapes[i].first}) << (i < shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < shapes.size(); j++) {
            result << ov::test::utils::vec2str(shapes[j].second[i]) << (j < shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "inputPRC=" << input_type.get_type_name() << "_";
    result << "targetPRC=" << convert_type.get_type_name() << "_";
    result << "specialValue=" << special_value_to_string(special_value) << "_";
    result << "trgDev=" << device_name;
    return result.str();
}

void ConversionLayerTest::SetUp() {
    ov::test::utils::ConversionTypes conversion_type;
    ov::element::Type convert_type;
    std::vector<InputShape> shapes;
    std::tie(conversion_type, shapes, input_type, convert_type, special_value, targetDevice) = GetParam();
    init_input_shapes(shapes);

    if (convert_type == ov::element::f16) {
        configuration.insert(ov::hint::inference_precision(ov::element::f16));
    }

    ov::ParameterVector params;
    for (const auto& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(input_type, shape));
    }

    std::shared_ptr<ov::Node> conversion;
    if (conversion_type == ov::test::utils::ConversionTypes::CONVERT) {
        conversion = std::make_shared<ov::op::v0::Convert>(params.front(), convert_type);
    } else /*CONVERT_LIKE*/ {
        auto like = std::make_shared<ov::op::v0::Constant>(convert_type, ov::Shape{1});
        conversion = std::make_shared<ov::op::v1::ConvertLike>(params.front(), like);
    }

    auto result = std::make_shared<ov::op::v0::Result>(conversion);
    function = std::make_shared<ov::Model>(result, params, "Conversion");
}

void ConversionLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (size_t i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        ov::Tensor tensor =
            ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
        if (special_value != ov::test::SpecialValue::none) {
            if (input_type == ov::element::f32) {
                modify_value<float>(tensor, special_value);
            } else if (input_type == ov::element::f16) {
                modify_value<ov::float16>(tensor, special_value);
            } else if (input_type == ov::element::bf16) {
                modify_value<ov::bfloat16>(tensor, special_value);
            } else if (input_type == ov::element::f8e4m3) {
                modify_value<ov::float8_e4m3>(tensor, special_value);
            } else if (input_type == ov::element::f8e5m2) {
                modify_value<ov::float8_e5m2>(tensor, special_value);
            }
        }

        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
}
}  // namespace test
}  // namespace ov
