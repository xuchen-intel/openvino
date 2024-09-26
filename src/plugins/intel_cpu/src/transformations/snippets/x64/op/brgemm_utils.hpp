// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "transformations/snippets/common/op/brgemm_common_utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"

namespace ov {
namespace intel_cpu {
namespace brgemm_utils {

dnnl::impl::cpu::x64::cpu_isa_t get_primitive_isa(const ov::element::Type& dt_in0, bool is_with_amx);

BRGEMM_TYPE get_brgemm_type(const element::Type& element_type_a, const Dimension& K_dim, const Dimension& N_dim, bool transpose_b);

/// \brief Computes number of elems with requested precision that fit in the vector register
size_t get_elems_in_vec(const ov::element::Type& precision);

/// \brief Computes VNNI factor used by OneDNN implementation. Depends on tensor precision
size_t compute_vnni_factor(const ov::element::Type& precision);

namespace repacking {
/**
 * @brief Computes buffer size that OneDNN impl needs for repacked tensor
 * @param copy_b_expr Repacking expression whose information (tensor precision, layout, subtensors) is used for
 * buffer size computations
 */
size_t get_repacking_buffer_size(const ov::snippets::lowered::ExpressionPtr& copy_b_expr);
}   // namespace repacking

}   // namespace brgemm_utils
}   // namespace intel_cpu
}   // namespace ov
