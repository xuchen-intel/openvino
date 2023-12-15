// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/snippets_isa.hpp"
#include "cpu_generator.hpp"

namespace ov {

class jit_snippet : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_snippet)

    ~jit_snippet() = default;

    jit_snippet() : jit_generator() {}

    void generate() override {}
};

intel_cpu::CompiledSnippetCPU::CompiledSnippetCPU(std::unique_ptr<dnnl::impl::cpu::aarch64::jit_generator> h) : h_compiled(std::move(h)) {
    OPENVINO_ASSERT(h_compiled && h_compiled->jit_ker(), "Got invalid jit generator or kernel was nopt compiled");
}

const uint8_t* intel_cpu::CompiledSnippetCPU::get_code() const {
    return h_compiled->jit_ker();
}

size_t intel_cpu::CompiledSnippetCPU::get_code_size() const {
    return h_compiled->getSize();
}

bool intel_cpu::CompiledSnippetCPU::empty() const {
    return get_code_size() == 0;
}

intel_cpu::CPUTargetMachine::CPUTargetMachine(dnnl::impl::cpu::aarch64::cpu_isa_t host_isa)
    : TargetMachine(), h(new jit_snippet()), isa(host_isa) {}

bool intel_cpu::CPUTargetMachine::is_supported() const {
    return dnnl::impl::cpu::aarch64::mayiuse(isa);
}

snippets::CompiledSnippetPtr intel_cpu::CPUTargetMachine::get_snippet() {
    if (h->create_kernel() != dnnl::impl::status::success) {
        OPENVINO_THROW("Failed to create jit_kernel in get_snippet()");
    }
    const auto& result = std::make_shared<CompiledSnippetCPU>(std::unique_ptr<dnnl::impl::cpu::aarch64::jit_generator>(h.release()));
    // Note that we reset all the generated code, since it was copied into CompiledSnippetCPU
    h.reset(new jit_snippet());
    return result;
}

size_t intel_cpu::CPUTargetMachine::get_lanes() const {
    switch (isa) {
        case dnnl::impl::cpu::aarch64::asimd : return dnnl::impl::cpu::aarch64::cpu_isa_traits<dnnl::impl::cpu::aarch64::asimd>::vlen / sizeof(float);
        default : OPENVINO_THROW("unknown isa ", isa);
    }
}

dnnl::impl::cpu::aarch64::cpu_isa_t intel_cpu::CPUTargetMachine::get_isa() const {
    return isa;
}

intel_cpu::CPUGenerator::CPUGenerator(dnnl::impl::cpu::aarch64::cpu_isa_t isa_) : Generator(std::make_shared<intel_cpu::CPUTargetMachine>(isa_)) {}

std::shared_ptr<snippets::Generator> intel_cpu::CPUGenerator::clone() const {
    const auto& cpu_target_machine = std::dynamic_pointer_cast<CPUTargetMachine>(target);
    OPENVINO_ASSERT(cpu_target_machine, "Failed to clone CPUGenerator: the instance contains incompatible TargetMachine type");
    return std::make_shared<CPUGenerator>(cpu_target_machine->get_isa());
}

snippets::Generator::opRegType intel_cpu::CPUGenerator::get_specific_op_reg_type(const std::shared_ptr<ov::Node>& op) const {
    // todo: add implementation
    OPENVINO_THROW("Register type of the operation " + std::string(op->get_type_name()) + " isn't determined!");
    return gpr2gpr;
}

bool intel_cpu::CPUGenerator::uses_precompiled_kernel(const std::shared_ptr<snippets::Emitter>& e) const {
    // todo: add implementation
    return false;
}

} // namespace ov