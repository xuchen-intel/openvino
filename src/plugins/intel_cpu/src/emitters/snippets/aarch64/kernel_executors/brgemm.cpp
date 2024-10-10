// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm.hpp"

#include "common/utils.hpp"
#include "dnnl_extension_utils.h"
#include "emitters/utils.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/common/op/brgemm_cpu.hpp"
#include "utils/general_utils.h"

#define DIM_CAST(X) static_cast<dnnl_dim_t>(X)
#define DTYPE_CAST(X) static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(X))

using namespace dnnl::impl;
using namespace ov::intel_cpu::brgemm_utils;

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

void BrgemmKernelConfig::update(dnnl_dim_t M, dnnl_dim_t N, dnnl_dim_t K, dnnl_dim_t LDA, dnnl_dim_t LDB, dnnl_dim_t LDC, float beta) {
    if (utils::one_of(0, M, N, K)) {
        m_M = 0; m_N = 0; m_K = 0;
        m_LDA = 0; m_LDB = 0; m_LDC = 0;
        m_beta = 0;
    } else {
        m_M = M; m_N = N; m_K = K;
        m_LDA = LDA; m_LDB = LDB; m_LDC = LDC;
        m_beta = beta;
    }
    m_hash = compute_hash();
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

    if (config.is_empty()) {
        std::cout << "****** early return" << std::endl;
        return compiled_kernel;
    }

    std::cout << "****** return" << std::endl;

    return compiled_kernel;
}

float BrgemmKernelExecutor::get_beta(const ov::snippets::lowered::LoopManagerPtr& loop_manager, int loop_id,
                                     const ov::snippets::lowered::ExpandedLoopInfoPtr& current_expanded_loop_info) {
    if (loop_id > 0) {
        const auto& current_unified_loop_info = current_expanded_loop_info->get_unified_loop_info();
        --loop_id;
        while (loop_id >= 0) {
            const auto& expanded_loop_info = loop_manager->get_loop_info<ov::snippets::lowered::ExpandedLoopInfo>(loop_id);
            if (expanded_loop_info->get_unified_loop_info() != current_unified_loop_info)
                return 0;
            if (expanded_loop_info->get_work_amount() > 0) {
                return 1;
            }
            --loop_id;
        }
    }
    return 0;
}

void BrgemmKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                         const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                         BrgemmKernelConfig& config) const {
    const auto& input_pds = expr->get_input_port_descriptors();
    const auto& output_pds = expr->get_output_port_descriptors();
    OV_CPU_JIT_EMITTER_ASSERT((input_pds.size() == 2 || input_pds.size() == 3) && output_pds.size() == 1,
                              "Invalid number of in/out port descriptors");

    const auto in0_shape = snippets::utils::get_planar_vdims(input_pds[0]->get_shape(), input_pds[0]->get_layout());
    const auto in1_shape = snippets::utils::get_planar_vdims(input_pds[1]->get_shape(), input_pds[1]->get_layout());
    auto in0_subtensor = input_pds[0]->get_subtensor();
    auto in1_subtensor = input_pds[1]->get_subtensor();

    auto M = *++in0_subtensor.rbegin();
    auto K = *in0_subtensor.rbegin();
    auto N = *in1_subtensor.rbegin();

    size_t loop_idx = 0;
    const auto& loop_ids = expr->get_loop_ids();
    const auto& loop_manager = linear_ir->get_loop_manager();
    auto get_loop_info = [&](){
        OPENVINO_ASSERT(loop_idx < loop_ids.size(), "Loop by dimension M is missed");
        return loop_manager->get_loop_info<ov::snippets::lowered::ExpandedLoopInfo>(loop_ids[loop_idx++]);
    };

    /* ------- Dimension M ----------*/
    if (ov::snippets::utils::is_full_dim_value(M)) {
        M = *++in0_shape.rbegin();
    } else {
        const auto& current_expanded_loop_info = get_loop_info();
        const auto& in_ports = current_expanded_loop_info->get_input_ports();
        const auto& out_ports = current_expanded_loop_info->get_output_ports();
        auto check_port = [&](const ov::snippets::lowered::LoopPort& p) { return p.dim_idx == 1; };
        OPENVINO_ASSERT(in_ports.size() > 1 && std::all_of(in_ports.cbegin(), in_ports.cend(), check_port) &&
                        out_ports.size() == 1 && check_port(out_ports.back()),
                        "Incorrect Loop by Brgemm dimension M");
        M = current_expanded_loop_info->get_increment();
        input_pds[0]->set_subtensor_dim(1, M);
        output_pds[0]->set_subtensor_dim(1, M);
    }

    /* ------- Dimension N ----------*/
    if (ov::snippets::utils::is_full_dim_value(N)) {
        N = *in1_shape.rbegin();
    } else {
        const auto& current_expanded_loop_info = get_loop_info();
        const auto& in_ports = current_expanded_loop_info->get_input_ports();
        const auto& out_ports = current_expanded_loop_info->get_output_ports();
        auto check_port = [&](const ov::snippets::lowered::LoopPort& p) { return p.dim_idx == 0; };
        OPENVINO_ASSERT(in_ports.size() == 2 && !in_ports.front().is_incremented && std::all_of(in_ports.cbegin(), in_ports.cend(), check_port) &&
                        out_ports.size() == 1 && check_port(out_ports.back()),
                        "Incorrect Loop by Brgemm dimension N");
        N = current_expanded_loop_info->get_increment();
        input_pds[1]->set_subtensor_dim(0, N);
        output_pds[0]->set_subtensor_dim(0, N);
    }

    /* ------- Dimension K ----------*/
    float beta = 0;
    if (ov::snippets::utils::is_full_dim_value(K)) {
        K = *in0_shape.rbegin();
    } else {
        const auto& current_expanded_loop_info = get_loop_info();
        const auto& in_ports = current_expanded_loop_info->get_input_ports();
        const auto& out_ports = current_expanded_loop_info->get_output_ports();
        OPENVINO_ASSERT(in_ports.size() == 2 && in_ports.front().dim_idx == 0 && in_ports.back().dim_idx == 1 &&
                        out_ports.size() == 1 && !out_ports.front().is_incremented,
                        "Incorrect Loop by Brgemm dimension K");
        K = current_expanded_loop_info->get_increment();
        input_pds[0]->set_subtensor_dim(0, K);
        input_pds[1]->set_subtensor_dim(1, K);
        if (K > 0)
            beta = get_beta(loop_manager, static_cast<int>(loop_ids.back()), current_expanded_loop_info);
    }

    const auto LDA = DIM_CAST(snippets::utils::get_dim_stride(expr->get_input_port(0)));
    const auto LDC = DIM_CAST(snippets::utils::get_dim_stride(expr->get_output_port(0)));
    auto LDB = DIM_CAST(snippets::utils::get_dim_stride(expr->get_input_port(1)));
    const auto& brgemm_node = as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(brgemm_node, "Got invalid node type in update_config");
    // In case of data repacking LDB is chosen in accordance with repacking buffer size
    if (with_repacking(brgemm_node->get_type()))
        LDB = brgemm_utils::repacking::compute_out_leading_dim(N, brgemm_node->get_input_element_type(1));

    config.update(DIM_CAST(M), DIM_CAST(N), DIM_CAST(K), LDA, LDB, LDC, beta);
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
