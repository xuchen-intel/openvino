// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_snippets_emitters.hpp"

#include <cpu/aarch64/jit_generator.hpp>
#include <cpu/aarch64/xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_adr.h>

#include "emitters/utils.hpp"

using namespace Xbyak_aarch64;

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

inline static void transform_idxs_to_regs(const std::vector<size_t>& idxs, std::vector<XReg>& regs) {
    regs.resize(idxs.size(), XReg(0));
    std::transform(idxs.begin(), idxs.end(), regs.begin(), [](size_t idx){return XReg(static_cast<uint32_t>(idx));});
}

NopEmitter::NopEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : aarch64::jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

jit_container_emitter::jit_container_emitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void jit_container_emitter::map_abstract_registers(mapping_info& gpr_map_pool,  mapping_info& vec_map_pool,
                                                   snippets::lowered::LinearIR::container& expressions) const {
    if (expressions.empty())
        OPENVINO_THROW("Cannot map registers when there is no allocated_emitters provided");
    auto map_regs = [](const std::vector<size_t>& abstract_regs, mapping_info& mapping) {
        auto& abstract_to_physical = mapping.first;
        auto& regs_pool = mapping.second;
        std::vector<size_t> physical_regs(abstract_regs.size());
        for (size_t i = 0; i < abstract_regs.size(); i++) {
            const auto abstract = abstract_regs[i];
            auto& physical = physical_regs[i];
            if (abstract_to_physical.count(abstract) == 0) {
                if (regs_pool.empty())
                    OPENVINO_THROW("Cannot map registers for jit_container_emitter: not enough regs in the pool");
                physical = regs_pool.back();
                regs_pool.pop_back();
                abstract_to_physical[abstract] = physical;
            } else {
                physical = abstract_to_physical[abstract];
            }
        }
        return physical_regs;
    };

    for (const auto& expression : expressions) {
        const auto& emitter = expression->get_emitter();
        std::vector<size_t> in_abstract_regs, out_abstract_regs;
        std::tie(in_abstract_regs, out_abstract_regs) = expression->get_reg_info();
        std::vector<size_t> in_physical_regs, out_physical_regs;
        switch (std::dynamic_pointer_cast<jit_emitter>(emitter)->get_in_out_type()) {
            case gpr_to_gpr:
                in_physical_regs = map_regs(in_abstract_regs, gpr_map_pool);
                out_physical_regs = map_regs(out_abstract_regs, gpr_map_pool);
                break;
            case gpr_to_vec:
                // Load Emitters
                in_physical_regs = map_regs(in_abstract_regs, gpr_map_pool);
                out_physical_regs = map_regs(out_abstract_regs, vec_map_pool);
                break;
            case vec_to_gpr:
                // Store Emitters
                in_physical_regs = map_regs(in_abstract_regs, vec_map_pool);
                out_physical_regs = map_regs(out_abstract_regs, gpr_map_pool);
                break;
            case vec_to_vec:
                // Regular operations
                in_physical_regs = map_regs(in_abstract_regs, vec_map_pool);
                out_physical_regs = map_regs(out_abstract_regs, vec_map_pool);
                break;
            default:
                OPENVINO_THROW("Unhandled in_out type");
        }
        expression->set_reg_info({in_physical_regs, out_physical_regs});
        if (auto container = std::dynamic_pointer_cast<jit_container_emitter>(expression->get_emitter()))
            container->map_abstract_registers(gpr_map_pool,  vec_map_pool, expressions);
    }
}

KernelEmitter::KernelEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr)
    : jit_container_emitter(h, isa, expr),
      reg_indexes_idx(0), // todo: revise
      reg_const_params_idx(1) { // todo: revise
    const auto kernel = ov::as_type_ptr<snippets::op::Kernel>(expr->get_node());
    if (!kernel)
        OPENVINO_THROW("KernelEmitter invoked with invalid op argument");
    if (kernel->region.empty())
        OPENVINO_THROW("KernelEmitter invoked with empty body");
    if (kernel->compile_params == nullptr)
        OPENVINO_THROW("KernelEmitter invoked with op::Kernel that contains no compile_params");
    body = kernel->region;
    jcp = *reinterpret_cast<const jit_snippets_compile_args*>(kernel->compile_params);
    master_shape = body.get_master_shape();
    // Note: plugin can prepend master shape with 1 to facilitate parallel execution (usually up to 6D tensor)
    //       so we have to reproduce this behavior here
    master_shape.insert(master_shape.begin(), jcp.parallel_executor_ndims - master_shape.size(), 1);
    const auto& io_exprs = body.get_IO_ops();
    num_inputs = 0;
    num_outputs = 0;
    for (const auto& expr : io_exprs) {
        snippets::lowered::PortDescriptorPtr desc = nullptr;
        element::Type etype;
        switch (expr->get_type()) {
            case snippets::lowered::IOExpression::io_type::INPUT: {
                const auto first_consumer = expr->get_output_port_connector(0)->get_consumers().begin()->get_expr();
                if (ov::is_type<snippets::op::RankNormalization>(first_consumer->get_node())) {
                    desc = first_consumer->get_output_port_descriptor(0);
                } else {
                    desc = expr->get_output_port_descriptor(0);
                }
                etype = expr->get_node()->get_output_element_type(0);
                num_inputs++;
                break;
            }
            case snippets::lowered::IOExpression::io_type::OUTPUT: {
                num_outputs++;
                desc = expr->get_input_port_descriptor(0);
                etype = expr->get_node()->get_input_element_type(0);
                break;
            } default : {
                OPENVINO_THROW("Kernel detected unsupported io_type");
            }
        }
        const auto& shape = desc->get_shape();
        const auto& layout = desc->get_layout();
        OPENVINO_ASSERT(shape.size() == layout.size(), "Shape and layout must have the same length");
        const auto max_dim = *std::max_element(layout.begin(), layout.end());
        OPENVINO_ASSERT(max_dim < shape.size(), "Max layout index can't be larger than the shape size");
        io_shapes.push_back(shape);
        io_data_layouts.push_back(layout);
        io_data_sizes.push_back(etype.size());
    }

    // Initialize pools of gp and vec registers
    gp_regs_pool.resize(16); // todo revise
    vec_regs_pool.resize(16); // todo revise
    // It's easier to remove the last item during mapping, so fill descending to map ascending
    for (size_t i = 0; i < 16; i++)
        gp_regs_pool[i] = vec_regs_pool[i] = 15 - i;
    // todo: it's more convenient to use std::set as a pool container (unique and always sorted),
    //  but pools are vectors to align with emit_code signature. Change signature?
    auto remove_regs_from_pool = [](std::vector<size_t>& pool, const std::set<size_t>& to_remove) {
        // It's important to keep the order of other elements
        pool.erase(std::remove_if(pool.begin(), pool.end(),
                                       [&](size_t x) {return to_remove.count(x) != 0;}), pool.end());
    };
    // Reserve stack base and pointer for push(...) and pop(...) operations
    // Reserve abi_param1 and abi_param2, since they'll be used to pass runtime call args to kernel
    remove_regs_from_pool(gp_regs_pool, {Xbyak::Operand::RSP, Xbyak::Operand::RBP,
                                         reg_indexes_idx, reg_const_params_idx});

    mapping_info gpr_map_pool({}, gp_regs_pool);
    mapping_info vec_map_pool({}, vec_regs_pool);
    snippets::lowered::LinearIR::container mem_access_exprs;
    snippets::lowered::LinearIR::container general_exprs;
    std::set<size_t> unique_buffers;

    for (const auto& expr : body) {
        // Brgemm is a special case since it incorporates input and output (we use onednn kernel)
        // Just like Load & Store it requires offsets calculation
        if (std::dynamic_pointer_cast<snippets::lowered::IOExpression>(expr)) {
            mem_access_exprs.emplace_back(expr);
        } else if (const auto buffer = ov::as_type_ptr<snippets::op::Buffer>(expr->get_node())) {
            const auto buffer_id = buffer->get_id();
            if (unique_buffers.count(buffer_id) == 0) {
                mem_access_exprs.push_back(expr);
                unique_buffers.insert(buffer_id);
            }
        } else {
            general_exprs.emplace_back(expr);
        }
    }
    num_unique_buffers = unique_buffers.size();

    // Note that we can't use reg_indexes_idx or reg_const_params_idx to store data pointers because these two
    // regs are used to calculate offsets for the data pointers
    map_abstract_registers(gpr_map_pool, vec_map_pool, mem_access_exprs);
    for (const auto& abstract_to_physical : gpr_map_pool.first)
        data_ptr_regs_idx.push_back(abstract_to_physical.second);
    // However we can use reg_indexes_idx and reg_const_params_idx for other operations since we won't need them
    // after offsets calculation
    gpr_map_pool.second.push_back(reg_indexes_idx);
    gpr_map_pool.second.push_back(reg_const_params_idx);
    map_abstract_registers(gpr_map_pool, vec_map_pool, general_exprs);
}

void KernelEmitter::emit_code(const std::vector<size_t> &in,
                              const std::vector<size_t> &out) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void KernelEmitter::validate_arguments(const std::vector<size_t> &in,
                                       const std::vector<size_t> &out) const {
    if (!in.empty())
        OPENVINO_THROW("KernelEmitter got invalid number of inputs. Expected 0, got ", in.size());
    if (!out.empty())
        OPENVINO_THROW("KernelEmitter got invalid number of outputs. Expected 0, got ", out.size());
    const auto num_params = num_inputs + num_outputs + num_unique_buffers;
    // The number of used gpr may be >= num_params since LoopBegin+LoopEnd could also use gpr to store work_amount
    if (data_ptr_regs_idx.size() != num_params)
        OPENVINO_THROW(
            "KernelEmitter: number of inputs and outputs is inconsistent with the number of allocated registers ",
            num_params,
            " data_ptr_regs_idx.size() = ",
            data_ptr_regs_idx.size());
}

void KernelEmitter::emit_impl(const std::vector<size_t>& in,
                              const std::vector<size_t>& out) const {
    h->preamble();

    XReg reg_indexes = XReg(static_cast<int>(reg_indexes_idx));
    XReg reg_const_params = XReg(static_cast<int>(reg_const_params_idx));
    std::vector<XReg> data_ptr_regs;
    transform_idxs_to_regs(data_ptr_regs_idx, data_ptr_regs);

    init_data_pointers(reg_indexes, reg_const_params, data_ptr_regs);
    for (const auto& expression : body) {
        const auto& emitter = expression->get_emitter();
        std::vector<size_t> in_regs, out_regs;
        std::tie(in_regs, out_regs) = expression->get_reg_info();
        emitter->emit_code(in_regs, out_regs, vec_regs_pool, gp_regs_pool);
    }
    h->postamble();
}

void KernelEmitter::init_data_pointers(const XReg& reg_indexes, const XReg& reg_const_params,
                                       const std::vector<XReg>& data_ptr_regs) const {
    const auto num_params = num_inputs + num_outputs;
    // Note that we don't need offset for the last dim, since it's handled directly by Tile emitter
    const size_t offset_rank = master_shape.size() - 1;
    std::vector<std::vector<size_t>> data_offsets(num_params, std::vector<size_t>{});
    auto offset_calculation = [=](const std::vector<size_t>& shape, const std::vector<size_t>& layout, const size_t data_size, bool is_input) {
        // Strides represent distance between consecutive elements of corresponding dimension.
        // If a dim size == 1, then the next dim starts immediately and the stride is 0
        // case 1:
        //    shape:         s0,    s1, s2, s3
        //    strides: s1*s2*s3, s2*s3, s3,  1
        // case 2:
        //    shape:      s0, s1, s2 == 1, s3
        //    strides: s1*s3, s3,       0,  1
        std::vector<size_t> strides(shape.size());
        size_t dim_step = 1;
        strides[shape.size() - 1] = 1;
        for (int k = static_cast<int>(shape.size()) - 2; k >= 0; k--) {
            dim_step *= shape[k+1];
            strides[k] = shape[k] != 1 ? dim_step * data_size : 0;
        }
        // Note: this is an extra copy, but let's keep it for clarity
        if (!layout.empty()) {
            std::vector<size_t> reordered_strides(strides.size());
            for (size_t i = 0; i < layout.size(); i++) {
                const auto& src_idx = is_input ? layout[i] : i;
                const auto& dst_idx = is_input ? i : layout[i];
                reordered_strides[dst_idx] = strides[src_idx];
            }
            strides = std::move(reordered_strides);
        }
        // the last stride is ignored, since the entire last dim is processed by kernel
        // and no parallel_for data_ptr offsets can be applied in this case
        strides.pop_back();
        // actual offset size might be larger that the shape size due to 6D scheduling
        strides.insert(strides.begin(), offset_rank - strides.size(), 0);

        return strides;
    };
    for (size_t i = 0; i < num_params; i++) {
        data_offsets[i] = offset_calculation(io_shapes[i],  io_data_layouts[i], io_data_sizes[i], i < num_inputs);
    }
    // master_shape size must be valid in both static and dynamic cases
    std::function<void(XReg, const std::vector<size_t>&, XReg, XReg)> init_ptr_with_offset;
    init_ptr_with_offset = [&](XReg pointer, const std::vector<size_t>& offsets, XReg reg_tmp, XReg reg_tmp1) {
        for (size_t j = 0; j < offset_rank; j++) {
            if (master_shape[j] != 1 && offsets[j] != 0) {
                h->mov(reg_tmp, offsets[j]);
                h->ldr(reg_tmp1, ptr(reg_indexes, static_cast<int32_t>(j * sizeof(size_t))));
                h->mul(reg_tmp, reg_tmp, reg_tmp1);
                h->add(pointer, pointer, reg_tmp);
            }
        }
    };
    const auto spare_corruptable_gpr = std::find_if(gp_regs_pool.begin(), gp_regs_pool.end(),
                                                   [this](size_t reg) {
                                                        return reg != reg_indexes_idx && reg != reg_const_params_idx;
                                                   });
    const bool last_iter_explicitly = spare_corruptable_gpr == gp_regs_pool.end();
    XReg reg_tmp = last_iter_explicitly ? data_ptr_regs[num_params - 1] : XReg(static_cast<uint32_t>(*spare_corruptable_gpr));
    XReg reg_tmp1 = XReg(30); // todo: revise
    // Vector "data_ptr_regs" is sorted by abstract regs.
    // It means that the vector contains the physical registers in order [src, .., src, dst, .., dst, buffer]
    // So we can initialize buffer register firstly as last value of vector "data_ptr_regs"
    // NOTE: Snippets Buffer Scratchpad has the common data pointer for all Buffers (even with different ID).
    //       The accessing memory is covered by correct offsets in each Buffer and the corresponding MemoryAccess ops
    for (size_t i = 0; i < num_unique_buffers; ++i) {
        h->ldr(data_ptr_regs[num_params + i], ptr(reg_const_params, static_cast<int32_t>(GET_OFF(buffer_scratchpad_ptr))));
    }
    size_t i = 0;
    for (; i < num_params - last_iter_explicitly; i++) {
        if (i < num_inputs)
            h->ldr(data_ptr_regs[i], ptr(reg_const_params, static_cast<int32_t>(GET_OFF(src_ptrs) + i * sizeof(void*))));
        else
            h->ldr(data_ptr_regs[i], ptr(reg_const_params, static_cast<int32_t>(GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*))));
        init_ptr_with_offset(data_ptr_regs[i], data_offsets[i], reg_tmp, reg_tmp1);
    }
    // a rare case when num_params is maximal, so we have no spare gprs
    // * Static case: we can use reg_const_params as the last reg_tmp for the last iteration (and corrupt it), since
    //     it won't be used anymore
    // * Dynamic case: we will need reg_const_params to pass runtime args to LoopScheduler, so we have to
    //     push a reg on the stack, and restore it value afterwards
    if (last_iter_explicitly) {
        h->ldr(data_ptr_regs[i], ptr(reg_const_params, static_cast<int32_t>(GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*))));
        reg_tmp = reg_const_params;
        // can corrupt reg_const_params, since we won't use it anymore
        init_ptr_with_offset(data_ptr_regs[i], data_offsets[i], reg_tmp, reg_tmp1);
    }
}

LoopBeginEmitter::LoopBeginEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : jit_emitter(h, isa) {
    loop_begin = ov::as_type_ptr<snippets::op::LoopBegin>(expr->get_node());
    if (!loop_begin)
        OPENVINO_THROW("LoopBeginEmitter invoked with invalid op argument");
    const auto& target_inputs = loop_begin->output(loop_begin->get_output_size() - 1).get_target_inputs();
    if (target_inputs.size() != 1)
        OPENVINO_THROW("LoopBeginEmitter invoked with invalid configuration: the last output must have exactly one "
                       "input attached");
    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(target_inputs.begin()->get_node()->shared_from_this());
    if (!loop_end)
        OPENVINO_THROW("LoopBeginEmitter invoked with invalid configuration: the last output must be LoopEnd");
    work_amount = loop_end->get_work_amount();
    evaluate_once = loop_end->get_evaluate_once();
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void LoopBeginEmitter::emit_code(const std::vector<size_t> &in,
                                 const std::vector<size_t> &out) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void LoopBeginEmitter::validate_arguments(const std::vector<size_t> &in,
                                          const std::vector<size_t> &out) const {
    if (!in.empty())
        OPENVINO_THROW("Invalid inputs size: expected 0 got ", in.size());
    if (out.size() != 1)
        OPENVINO_THROW("Invalid outputs size: expected 1 got ", out.size());
}

void LoopBeginEmitter::emit_impl(const std::vector<size_t>& in,
                                 const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in, out);
    } else {
        OPENVINO_THROW("LoopBegin emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void LoopBeginEmitter::emit_isa(const std::vector<size_t>& in,
                                const std::vector<size_t>& out) const {
    XReg reg_work_amount = XReg(out[0]);

    // save previous register state (if there is an outer loop that uses this reg for example)
    if (!evaluate_once) {
        h->mov(reg_work_amount, work_amount);
    }
    // Note: loop address is not calculated at this point, so need to call calcJmpAddress() which is protected
    // or ready(), but they both set internal flags and that's not a desired way to use them.
    // So the most obvious WA is just to use current address manually
    loop_begin->begin_address = h->getCurr();
}

LoopEndEmitter::LoopEndEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : jit_emitter(h, isa) {
    loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(expr->get_node());
    if (!loop_end)
        OPENVINO_THROW("LoopEndEmitter invoked with invalid op argument");
    loop_begin = loop_end->get_loop_begin();
    if (!loop_begin)
        OPENVINO_THROW("LoopEndEmitter invoked with invalid configuration: the last arg must be LoopBegin");
    num_inputs = loop_end->get_input_num();
    num_outputs = loop_end->get_output_num();
    wa_increment = static_cast<int64_t>(loop_end->get_increment());
    work_amount = static_cast<int64_t>(loop_end->get_work_amount());
    ptr_increments = loop_end->get_ptr_increments();
    finalization_offsets = loop_end->get_finalization_offsets();
    evaluate_once = loop_end->get_evaluate_once();
    io_data_size = loop_end->get_element_type_sizes();
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void LoopEndEmitter::emit_code(const std::vector<size_t> &in,
                               const std::vector<size_t> &out) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void LoopEndEmitter::validate_arguments(const std::vector<size_t> &in,
                                        const std::vector<size_t> &out) const {
    if (out.size() != num_outputs)
        OPENVINO_THROW("Invalid number of out arguments: expected ", num_outputs, " got ", out.size());
    if (in.size() != num_inputs)
        OPENVINO_THROW("Invalid number of in arguments: expected ", num_inputs , " got ", in.size());
    const auto io_size = num_inputs - 1;
    if (ptr_increments.size() != io_size)
        OPENVINO_THROW("Invalid ptr_increments size: expected ", io_size, " got ", ptr_increments.size());
    if (finalization_offsets.size() != io_size)
        OPENVINO_THROW("Invalid finalization_offsets size: expected: ", io_size, " got ", finalization_offsets.size());
}

void LoopEndEmitter::emit_impl(const std::vector<size_t>& in,
                               const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in, out);
    } else {
        OPENVINO_THROW("LoopEnd emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void LoopEndEmitter::emit_isa(const std::vector<size_t>& in,
                              const std::vector<size_t>& out) const {
    std::vector<size_t> data_ptr_reg_idxs;
    // the last input is actually a work_amount reg
    data_ptr_reg_idxs.reserve(num_inputs - 1);
    std::copy(in.begin(), in.end() - 1, std::back_inserter(data_ptr_reg_idxs));
    std::vector<XReg> data_ptr_regs;
    transform_idxs_to_regs(data_ptr_reg_idxs, data_ptr_regs);
    XReg reg_work_amount = XReg(in.back());
    if (!evaluate_once) {
        for (size_t idx = 0; idx < data_ptr_regs.size(); idx++) {
            if (ptr_increments[idx] != 0)
                h->add(data_ptr_regs[idx], data_ptr_regs[idx], ptr_increments[idx] * wa_increment * io_data_size[idx]);
        }
        h->sub(reg_work_amount, reg_work_amount, wa_increment);
        h->cmp(reg_work_amount, wa_increment);
        h->b(GE, reinterpret_cast<int64_t>(loop_begin->begin_address) - reinterpret_cast<int64_t>(h->getCurr()));
    }

    for (size_t idx = 0; idx < data_ptr_regs.size(); idx++) {
        if (finalization_offsets[idx] != 0)
            h->add(data_ptr_regs[idx], data_ptr_regs[idx], finalization_offsets[idx] * io_data_size[idx]);
    }
}

MemoryEmitter::MemoryEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : jit_emitter(h, isa) {
    const auto n = expr->get_node();
    src_prc = n->get_input_element_type(0);
    dst_prc = n->get_output_element_type(0);
}

LoadEmitter::LoadEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : MemoryEmitter(h, isa, expr) {
    if (src_prc != dst_prc)
        OPENVINO_THROW("LoadEmitter supports only equal input and output types but gets: ",
                       src_prc.get_type_name(),
                       " and ",
                       dst_prc.get_type_name());

    const auto load = std::dynamic_pointer_cast<snippets::op::Load>(expr->get_node());
    count = load->get_count();
    byte_offset = load->get_offset();
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
    load_emitter.reset(new jit_load_emitter(h, isa, src_prc, dst_prc, count));
}

void LoadEmitter::emit_impl(const std::vector<size_t>& in,
                            const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in, out);
    } else {
        OPENVINO_THROW("Load emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void LoadEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    if (!load_emitter)
        OPENVINO_THROW("Load CPU emitter isn't initialized for LoadEmitter!");
    load_emitter->emit_code({in[0], byte_offset}, {out[0]}, convert_to_size_t<uint32_t>(aux_vec_idxs),
                            convert_to_size_t<uint32_t>(aux_gpr_idxs));
}

void LoadEmitter::emit_data() const {
    load_emitter->emit_data();
}

StoreEmitter::StoreEmitter(jit_generator* h, cpu_isa_t isa, const ExpressionPtr& expr) : MemoryEmitter(h, isa, expr) {
    if (src_prc != dst_prc)
        OPENVINO_THROW("StoreEmitter supports only equal input and output types but gets: ",
                       src_prc.get_type_name(),
                       " and ",
                       dst_prc.get_type_name());

    const auto store = ov::as_type_ptr<snippets::op::Store>(expr->get_node());
    count = store->get_count();
    byte_offset = store->get_offset();
    in_out_type_ = emitter_in_out_map::vec_to_gpr;
    store_emitter.reset(new jit_store_emitter(h, isa, src_prc, dst_prc, count));
}

void StoreEmitter::emit_impl(const std::vector<size_t>& in,
                             const std::vector<size_t>& out) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in, out);
    } else {
        OPENVINO_THROW("Store emitter doesn't support ", host_isa_);
    }
}

template <cpu_isa_t isa>
void StoreEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    if (!store_emitter)
        OPENVINO_THROW("Store CPU emitter isn't initialized for StoreEmitter!");
    store_emitter->emit_code({in[0], byte_offset}, {out[0]}, convert_to_size_t<uint32_t>(aux_vec_idxs),
                             convert_to_size_t<uint32_t>(aux_gpr_idxs));
}

void StoreEmitter::emit_data() const {
    store_emitter->emit_data();
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
