// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm.hpp"

#include "common/utils.hpp"
#include "dnnl_extension_utils.h"
#include "utils/general_utils.h"

#define DTYPE_CAST(X) static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(X))

using namespace dnnl::impl;

namespace {
size_t init_hash(dnnl_data_type_t dt_in0, dnnl_data_type_t dt_in1, dnnl::impl::cpu::aarch64::cpu_isa_t isa) {
    size_t seed = 0;
#define HASH(X) seed = hash_combine(seed, X)
    HASH(dt_in0); HASH(dt_in1);
    HASH(isa);
#undef HASH
    return seed;
}
} // namespace

namespace ov {
namespace intel_cpu {
namespace aarch64 {

BrgemmKernelConfig::BrgemmKernelConfig(const element::Type& in0_dtype, const element::Type& in1_dtype,
                                       dnnl::impl::cpu::aarch64::cpu_isa_t primitive_isa) :
                                       m_static_params(std::make_shared<StaticParams>(in0_dtype, in1_dtype, primitive_isa)) {
    m_hash = compute_hash();
}

bool BrgemmKernelConfig::is_completed() const {
    return !utils::one_of(0, m_M, m_N, m_K, m_LDA, m_LDB, m_LDC) || is_empty();
}

bool BrgemmKernelConfig::is_empty() const {
    return everyone_is(0, m_M, m_N, m_K, m_LDA, m_LDB, m_LDC, m_beta);
}

BrgemmKernelConfig::StaticParams::StaticParams(const element::Type& in0_dtype, const element::Type& in1_dtype,
                                               dnnl::impl::cpu::aarch64::cpu_isa_t primitive_isa) :
                                               dt_in0(DTYPE_CAST(in0_dtype)), dt_in1(DTYPE_CAST(in1_dtype)),
                                               isa(primitive_isa),
                                               hash(init_hash(dt_in0, dt_in1, isa)) {
}

size_t BrgemmKernelConfig::compute_hash() const {
    size_t seed = m_static_params->hash;
#define HASH(X) seed = hash_combine(seed, X)
    HASH(m_M); HASH(m_N); HASH(m_K);
    HASH(m_LDA); HASH(m_LDB); HASH(m_LDC);
    HASH(m_beta);
#undef HASH
    return seed;
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
