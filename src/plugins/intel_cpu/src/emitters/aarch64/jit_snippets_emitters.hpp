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

///
/// Memory emitters:
///
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

    // todo: revise parameters
    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;
    void emit_data() const override;

private:
    std::unique_ptr<jit_load_emitter> load_emitter = nullptr;
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

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
