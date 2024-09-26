// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_common_utils.hpp"

#include "dnnl_extension_utils.h"
#include "emitters/utils.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/common/op/brgemm_copy_b.hpp"
#include "utils/general_utils.h"

using namespace dnnl::impl;
using namespace ov::snippets::utils;

namespace ov {
namespace intel_cpu {
namespace brgemm_utils {

namespace repacking {

size_t get_compensations_buffer_size(const ov::snippets::lowered::ExpressionPtr& copy_b_expr) {
    OPENVINO_ASSERT(ov::is_type<ov::intel_cpu::BrgemmCopyB>(copy_b_expr->get_node()));
    const auto& in_subtensor = ov::snippets::utils::get_projected_subtensor(copy_b_expr->get_input_port(0));
    const size_t n_blk = *in_subtensor.rbegin();
    OPENVINO_ASSERT(!is_dynamic_value(n_blk), "get_compensations_buffer_size must be called with static subtensor values");
    const auto& precision = copy_b_expr->get_node()->get_input_element_type(0);
    // Compensations are computed during repacking, so we need to round-up allocation shape according to m_inner_n_block
    // because of OneDNN implementation nuances (as in get_repacking_buffer_size).
    // However, the compensations are computed by N dimension, so K dimension doesn't affect the compensations buffer
    return std::max(n_blk, compute_inner_n_block(precision));
}

size_t compute_out_leading_dim(const size_t n_block, const ov::element::Type& precision) {
    return std::max(n_block, compute_inner_n_block(precision));
}

size_t compute_inner_n_block(const ov::element::Type& precision) {
    switch (precision) {
        case element::i8: return 64;
        case element::bf16: return 32;
        case element::f32: return 16;
        default: OPENVINO_THROW("BrgemmCopyB doesn't support precision ", precision);
    }
}
}   // namespace repacking
}   // namespace brgemm_utils
}   // namespace intel_cpu
template <>
EnumNames<ov::intel_cpu::BRGEMM_TYPE>& EnumNames<ov::intel_cpu::BRGEMM_TYPE>::get() {
    static auto enum_names =
            EnumNames<ov::intel_cpu::BRGEMM_TYPE>("ov::intel_cpu::jit_bgremm_utils::BRGEMM_TYPE",
                                                                {{"stand_alone", ov::intel_cpu::BRGEMM_TYPE::STAND_ALONE},
                                                                 {"with_amx", ov::intel_cpu::BRGEMM_TYPE::WITH_AMX},
                                                                 {"with_compensations", ov::intel_cpu::BRGEMM_TYPE::WITH_COMPENSATIONS},
                                                                 {"repacking_only", ov::intel_cpu::BRGEMM_TYPE::REPACKING_ONLY}});
    return enum_names;
}
}   // namespace ov
