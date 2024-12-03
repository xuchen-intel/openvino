// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API FakeConvertDecomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief FakeConvertDecomposition transformation decomposes FakeConvert layer.
 *
 * output = (convert_f8_to_f32(convert_f32_to_f8(input * scale - shift)) + shift) / scale
 *
 */

class ov::pass::FakeConvertDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FakeConvertDecomposition", "0");
    FakeConvertDecomposition();
};
