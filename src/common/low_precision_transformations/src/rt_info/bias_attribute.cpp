// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/bias_attribute.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <iterator>
#include <vector>

void ov::mark_as_bias(const std::shared_ptr<ov::Node>& node) {
    auto& rt = node->get_rt_info();
    rt[ov::BiasAttribute::get_type_info_static()] = ov::BiasAttribute();
}

bool ov::marked_as_bias(const std::shared_ptr<const ov::Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.find(ov::BiasAttribute::get_type_info_static()) != rt_info.end();
}
