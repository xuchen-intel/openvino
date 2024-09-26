// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/aarch64/brgemm/brgemm.hpp>
#include "emitters/snippets/cpu_kernel_executor_table.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

struct BrgemmKernelConfig : public snippets::KernelExecutorBase::GenericConfig {
public:
    BrgemmKernelConfig(const element::Type& in0_dtype, const element::Type& in1_dtype, dnnl::impl::cpu::aarch64::cpu_isa_t primitive_isa);
    BrgemmKernelConfig() = delete;
    bool is_completed() const override;
    size_t hash() const override { return m_hash; }
    bool operator==(const BrgemmKernelConfig& rhs) const;
    bool operator!=(const BrgemmKernelConfig& rhs) const {return !(*this == rhs);}
    std::unique_ptr<GenericConfig> get_clone_ptr() const override {
        return std::unique_ptr<BrgemmKernelConfig>( new BrgemmKernelConfig(*this));
    }
    bool is_empty() const;

private:
    struct StaticParams {
        StaticParams(const element::Type& in0_dtype, const element::Type& in1_dtype, dnnl::impl::cpu::aarch64::cpu_isa_t primitive_isa);
        const dnnl_data_type_t dt_in0 {dnnl_f32}, dt_in1 {dnnl_f32};
        const dnnl::impl::cpu::aarch64::cpu_isa_t isa {dnnl::impl::cpu::aarch64::isa_undef};
        const size_t hash {0};
        bool operator==(const StaticParams& rhs) const;
        bool operator!=(const StaticParams& rhs) const { return !(*this == rhs); }
    };
    size_t compute_hash() const;
    std::shared_ptr<StaticParams> m_static_params;
    dnnl_dim_t m_M {0}, m_N {0}, m_K {0}, m_LDA {0}, m_LDB {0}, m_LDC {0};
    float m_beta {0};
    size_t m_hash {SIZE_MAX};
};

struct BrgemmCompiledKernel {
    std::unique_ptr<dnnl::impl::cpu::aarch64::brgemm_kernel_t> compiled_kernel = nullptr;
    // Note: Palette is treated as a part of a kernel because it is initialized during the kernel compilation stage.
    //       Each kernel need to store the pallet it was compiled with.
    char palette[64] = {};
};

class BrgemmKernelExecutor : public CPUKernelExecutor<BrgemmKernelConfig, BrgemmCompiledKernel> {
public:
    BrgemmKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmKernelConfig config);

protected:
    std::shared_ptr<BrgemmCompiledKernel> compile_kernel(const BrgemmKernelConfig& c) const override;
    void update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                       const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                       BrgemmKernelConfig& config) const override;
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
