
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/aarch64/jit_emitter.hpp"
#include "jit_container_emitter.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

#define SNIPPETS_MAX_SNIPPETS_DIMS 12
#define GET_OFF(field) offsetof(jit_snippets_call_args, field)

struct jit_snippets_call_args {
    const void *src_ptrs[SNIPPETS_MAX_SNIPPETS_DIMS] = {};
    void *dst_ptrs[SNIPPETS_MAX_SNIPPETS_DIMS] = {};
    void *buffer_scratchpad_ptr = nullptr;
};

struct jit_snippets_compile_args {
    size_t parallel_executor_ndims = 1;
};

/// \brief  Kernel is the only entry point to Codogen Jit compilation. Kernel perform abstract-to-physical register
/// mapping and creates a pools of available gpr and vec registers. Kernel usually contains (at least one)
/// LoopBeginEmitter and LoopEndEmitter pair. In general the enclosed emitters should be organized in the following way:
/// KernelEmitter {                 /* entry point, maps registers, creates pools of available registers */
///     1.S LoopBeginEmitter        /* Scalar Loop over the outer dimension [START] */
///         2.S LoopBeginEmitter    /* inner vector loop [START] */
///             ...                 /* All the necessary Load/Strore/elementwise emitters */
///         2.E LoopEndEmitter      /* inner vector loop [END] */
///         3.S LoopBeginEmitter    /* inner scalar loop for tail processing [START]*/
///             ...                 /* All the necessary Load/Strore/elementwise emitters */
///         3.E LoopEndEmitter      /* inner scalar loop for tail processing [END]*/
///     1.E LoopEndEmitter          /* Scalar Loop over the outer dimension [END] */
/// }
/// Note that Kernel doesn't accept any input arguments.
class KernelEmitter : public jit_container_emitter {
public:
    KernelEmitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                  dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                  const ov::snippets::lowered::ExpressionPtr& expr);

    void emit_code(const std::vector<size_t> &in,
                   const std::vector<size_t> &out) const;
    size_t get_inputs_count() const override {return 0;}

private:
    using jit_emitter::emit_code;
    void validate_arguments(const std::vector<size_t> &in,
                            const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;
    void init_data_pointers(const Xbyak_aarch64::XReg&, const Xbyak_aarch64::XReg&, const std::vector<Xbyak_aarch64::XReg>&) const;

    jit_snippets_compile_args jcp;
    std::vector<size_t> gp_regs_pool;
    std::vector<size_t> master_shape;
    size_t num_inputs;
    size_t num_outputs;
    size_t num_unique_buffers;
    // Vector of indices (length = input tensor rank) per every input and output that describes in which order
    // corresponding tensor dimensions are accessed (default: consecutive dense, e.g. 0,1,2,3 for 4D tensor).
    // Needed to calc i/o offsets.
    std::vector<std::vector<size_t>> io_data_layouts;
    std::vector<std::vector<size_t>> io_shapes = {};
    std::vector<size_t> io_data_sizes {};

    // gpr's used to store data pointers, track them to apply offsets in Kernel
    std::vector<size_t> data_ptr_regs_idx;
    std::vector<size_t> vec_regs_pool;

    const size_t reg_indexes_idx;
    const size_t reg_const_params_idx;
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
