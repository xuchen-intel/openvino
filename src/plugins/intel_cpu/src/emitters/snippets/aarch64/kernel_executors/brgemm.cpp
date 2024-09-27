// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm.hpp"

#include "common/utils.hpp"
#include "dnnl_extension_utils.h"
#include "utils/general_utils.h"
#include "emitters/utils.hpp"

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

bool BrgemmKernelConfig::operator==(const BrgemmKernelConfig& rhs) const {
#define EQ(X) X == rhs.X
    return EQ(m_hash) && EQ(m_beta) &&
           EQ(m_M) && EQ(m_N) && EQ(m_K) &&
           EQ(m_LDA) && EQ(m_LDB) && EQ(m_LDC) &&
           (EQ(m_static_params.get()) || *m_static_params == *(rhs.m_static_params));
#undef EQ
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

bool BrgemmKernelConfig::StaticParams::operator==(const StaticParams& rhs) const {
#define EQ(X) X == rhs.X
    return EQ(hash) && EQ(dt_in0) && EQ(dt_in1) && EQ(isa);
#undef EQ
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

BrgemmKernelExecutor::BrgemmKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmKernelConfig config) :
    CPUKernelExecutor<BrgemmKernelConfig, BrgemmCompiledKernel>(std::move(kernel_cache), std::move(config)) {
}

std::shared_ptr<BrgemmCompiledKernel> BrgemmKernelExecutor::compile_kernel(const BrgemmKernelConfig& config) const {
    std::shared_ptr<BrgemmCompiledKernel> compiled_kernel = std::make_shared<BrgemmCompiledKernel>();

    return compiled_kernel;
}

void BrgemmKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                         const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                         BrgemmKernelConfig& config) const {
}

void BrgemmKernelExecutor::execute(const BrgemmKernelExecutor* executor, call_args* args) {
    std::cout << "###### BrgemmKernelExecutor::execute ######" << std::endl;

    auto kernel = executor->get_kernel();
    const auto& config = static_cast<const BrgemmKernelConfig&>(executor->get_config());
    OV_CPU_JIT_EMITTER_ASSERT(kernel, "has nullptr compiler kernel or invalid config");

    cpu::aarch64::brgemm_kernel_params_t brgemm_p;

    brgemm_p.batch = nullptr;  // default value
    brgemm_p.ptr_A = args->A;
    brgemm_p.ptr_B = args->B;
    brgemm_p.ptr_C = args->C;
    brgemm_p.ptr_D = args->C;
    brgemm_p.ptr_buf = args->scratch;
    brgemm_p.ptr_bias = nullptr;
    brgemm_p.do_post_ops = 0;
    brgemm_p.do_apply_comp = 0;
    brgemm_p.skip_accm = 0;
    brgemm_p.BS = 1;  // default value
    OV_CPU_JIT_EMITTER_ASSERT(kernel->compiled_kernel, "has nullptr kernel");
    (*kernel->compiled_kernel)(&brgemm_p);
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
