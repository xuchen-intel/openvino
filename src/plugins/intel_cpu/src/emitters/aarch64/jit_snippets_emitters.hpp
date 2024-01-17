// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_emitter.hpp"
#include "jit_load_store_emitters.hpp"

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

class NopEmitter : public jit_emitter {
public:
    NopEmitter(dnnl::impl::cpu::aarch64::jit_generator* h,
               dnnl::impl::cpu::aarch64::cpu_isa_t isa,
               const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override {}
};

/// \brief jit_container_emitter designed to wrap Emitters that contain other Emitters (for example, KernelEmitter).
///  This is needed to provide common interface for register mapping
/// (abstract to physical) and nested code access.
class jit_container_emitter: public jit_emitter {
public:
    jit_container_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                          dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                          const ov::snippets::lowered::ExpressionPtr& expr);
    // mapping info contains abstract_to_physical map + regs_pool
    using mapping_info = std::pair<std::map<size_t, size_t>, std::vector<size_t>&>;
protected:
    // maps gpr and vec abstract registers to physical ones.
    void map_abstract_registers(mapping_info& gpr_map_pool,  mapping_info& vec_map_pool,
                                snippets::lowered::LinearIR::container& expressions) const;
    snippets::lowered::LinearIR body;
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

class LoopBeginEmitter : public jit_emitter {
public:
    LoopBeginEmitter(dnnl::impl::cpu::aarch64::jit_generator* h,
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
    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in,
                  const std::vector<size_t>& out) const;

    std::shared_ptr<snippets::op::LoopBegin> loop_begin;
    bool evaluate_once = false;
    size_t work_amount = 0;
};

class LoopEndEmitter : public jit_emitter {
public:
    LoopEndEmitter(dnnl::impl::cpu::aarch64::jit_generator* h,
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
    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in,
                  const std::vector<size_t>& out) const;

    std::shared_ptr<snippets::op::LoopBegin> loop_begin;
    std::shared_ptr<snippets::op::LoopEnd> loop_end;

    size_t num_inputs = 0;
    size_t num_outputs = 0;
    // keep data_size int64_t to avoid conversion to size_t (and overflow) when multiplied by negative increments or offsets
    std::vector<int64_t> io_data_size {};
    int64_t wa_increment = 0;
    int64_t work_amount = 0;
    bool evaluate_once = false;
    std::vector<int64_t> ptr_increments;
    std::vector<int64_t> finalization_offsets;
};

/// \brief MemoryEmitter is the base class for emitters with load/store functionality.
/// *Note*: post increment is embedded into Load/Store operation which means that
/// it's illigal to load/store to the same address multiple times
/// Typical application can be if Load and BroadcastLoad are performed from the same pointer.
/// If Load goes before BroadcastLoad topologicaly the resilt will be incorrect
/// For scalar loads we can use different loops. Tiling indeed can be arbitrary and post increment should be somehow coded into ISA.
/// Blocked parameter to tell if input is actually blocked. Broadcast means broadcast by W in other cases no need to substitute load.
class MemoryEmitter : public jit_emitter  {
public:
    MemoryEmitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                  dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                  const ov::snippets::lowered::ExpressionPtr& expr);

protected:
    ov::element::Type src_prc;
    ov::element::Type dst_prc;

    size_t count = 0;
    size_t byte_offset = 0;
};

class LoadEmitter : public MemoryEmitter {
public:
    LoadEmitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
    void emit_data() const override;

private:
    std::unique_ptr<jit_load_emitter> load_emitter = nullptr;
};

class BroadcastLoadEmitter : public MemoryEmitter {
public:
    BroadcastLoadEmitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                         dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                         const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {return 0;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
};

class StoreEmitter : public MemoryEmitter  {
public:
    StoreEmitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                 dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                 const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
    void emit_data() const override;

private:
    std::unique_ptr<jit_store_emitter> store_emitter = nullptr;
};

class ScalarEmitter : public jit_emitter {
public:
    ScalarEmitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                  dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                  const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {return 0;}

protected:
    size_t get_aux_gprs_count() const override {return 1;}

private:
    void emit_impl(const std::vector<size_t>& in,
              const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;

private:
    int32_t value;
};

class FillEmitter : public jit_emitter {
public:
    FillEmitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {return 1;}

protected:
    size_t get_aux_gprs_count() const override;

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void fill_full(const std::vector<size_t> &out) const;
    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void fill_tail(const std::vector<size_t> &in, const std::vector<size_t> &out) const;

    bool is_full_reg() const { return offset == 0; }
    bool is_optimized() const { return is_full_reg() && fill_value == uint32_t(0x0); }

    size_t offset = 0;
    uint32_t fill_value = 0x0;
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
