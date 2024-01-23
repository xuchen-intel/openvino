// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/snippets_isa.hpp"
#include "cpu_generator.hpp"
#include "jit_snippets_emitters.hpp"
#include "jit_eltwise_emitters.hpp"

#include <openvino/opsets/opset13.hpp>

namespace ov {

#define CREATE_SNIPPETS_EMITTER(e_type) { \
    [this](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
        return std::make_shared<e_type>(h.get(), isa, expr); \
    }, \
    [](const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> { \
        return e_type::get_supported_precisions(n); \
    } \
}

#define CREATE_CPU_EMITTER(e_type) { \
    [this](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
        return std::make_shared<e_type>(h.get(), isa, expr->get_node()); \
    }, \
    [](const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> { \
        return e_type::get_supported_precisions(n); \
    } \
}

class jit_snippet : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_snippet)

    ~jit_snippet() = default;

    jit_snippet() : jit_generator() {}

    void generate() override {}
};

namespace intel_cpu {
namespace aarch64 {

CompiledSnippetCPU::CompiledSnippetCPU(std::unique_ptr<dnnl::impl::cpu::aarch64::jit_generator> h) : h_compiled(std::move(h)) {
    OPENVINO_ASSERT(h_compiled && h_compiled->jit_ker(), "Got invalid jit generator or kernel was nopt compiled");
}

const uint8_t* CompiledSnippetCPU::get_code() const {
    return h_compiled->jit_ker();
}

size_t CompiledSnippetCPU::get_code_size() const {
    return h_compiled->getSize();
}

bool CompiledSnippetCPU::empty() const {
    return get_code_size() == 0;
}

CPUTargetMachine::CPUTargetMachine(dnnl::impl::cpu::aarch64::cpu_isa_t host_isa)
    : TargetMachine(), h(new jit_snippet()), isa(host_isa) {
    // data movement
    jitters[op::v0::Parameter::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(NopEmitter);
    jitters[op::v0::Result::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(NopEmitter);

    // memory access
    jitters[snippets::op::Load::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(LoadEmitter);
    jitters[snippets::op::Store::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(StoreEmitter);

    // binary
    jitters[op::v1::Add::get_type_info_static()] = CREATE_CPU_EMITTER(jit_add_emitter);

    // control flow
    jitters[snippets::op::LoopBegin::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(LoopBeginEmitter);
    jitters[snippets::op::LoopEnd::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(LoopEndEmitter);
}

bool CPUTargetMachine::is_supported() const {
    return dnnl::impl::cpu::aarch64::mayiuse(isa);
}

snippets::CompiledSnippetPtr CPUTargetMachine::get_snippet() {
    if (h->create_kernel() != dnnl::impl::status::success) {
        OPENVINO_THROW("Failed to create jit_kernel in get_snippet()");
    }
    const auto& result = std::make_shared<CompiledSnippetCPU>(std::unique_ptr<dnnl::impl::cpu::aarch64::jit_generator>(h.release()));
    // Note that we reset all the generated code, since it was copied into CompiledSnippetCPU
    h.reset(new jit_snippet());
    return result;
}

size_t CPUTargetMachine::get_lanes() const {
    switch (isa) {
        case dnnl::impl::cpu::aarch64::asimd : return dnnl::impl::cpu::aarch64::cpu_isa_traits<dnnl::impl::cpu::aarch64::asimd>::vlen / sizeof(float);
        default : OPENVINO_THROW("unknown isa ", isa);
    }
}

dnnl::impl::cpu::aarch64::cpu_isa_t CPUTargetMachine::get_isa() const {
    return isa;
}

CPUGenerator::CPUGenerator(dnnl::impl::cpu::aarch64::cpu_isa_t isa_) : Generator(std::make_shared<CPUTargetMachine>(isa_)) {}

std::shared_ptr<snippets::Generator> CPUGenerator::clone() const {
    const auto& cpu_target_machine = std::dynamic_pointer_cast<CPUTargetMachine>(target);
    OPENVINO_ASSERT(cpu_target_machine, "Failed to clone CPUGenerator: the instance contains incompatible TargetMachine type");
    return std::make_shared<CPUGenerator>(cpu_target_machine->get_isa());
}

snippets::Generator::opRegType CPUGenerator::get_specific_op_reg_type(const std::shared_ptr<ov::Node>& op) const {
    // todo: add implementation
    OPENVINO_THROW("Register type of the operation " + std::string(op->get_type_name()) + " isn't determined!");
    return gpr2gpr;
}

bool CPUGenerator::uses_precompiled_kernel(const std::shared_ptr<snippets::Emitter>& e) const {
    // todo: add implementation
    return false;
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
