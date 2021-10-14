// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_topk_node.h"

#include <mkldnn.hpp>

#include <string>
#include <vector>
#include <set>
#include <mkldnn_extension_utils.h>
#include "emitters/jit_load_store_emitters.hpp"
#include "ie_parallel.hpp"
#include <ngraph/op/topk.hpp>
#include <algorithm>

#include <cpu/x64/jit_generator.hpp>
#include <cpu/x64/jit_uni_eltwise.hpp>
#include "common/cpu_memcpy.h"

#include <ngraph/opsets/opset1.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu::x64;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_topk_call_args, field)

#define vmm_mask    Vmm(0)
#define vmm_tmp     Vmm(1)
#define vmm_val(i)  Vmm(2 * (i) + 2)
#define vmm_idx(i)  Vmm(2 * (i) + 3)
#define vmm_val_l   Vmm(2)
#define vmm_idx_l   Vmm(3)
#define vmm_val_r   Vmm(4)
#define vmm_idx_r   Vmm(5)

#define xmm_mask    Xmm(0)
#define xmm_tmp     Xmm(1)
#define xmm_val(i)  Xmm(2 * (i) + 2)
#define xmm_idx(i)  Xmm(2 * (i) + 3)
#define xmm_val_l   Xmm(2)
#define xmm_idx_l   Xmm(3)
#define xmm_val_r   Xmm(4)
#define xmm_idx_r   Xmm(5)

#define xmm_val_p   Xmm(6)
#define xmm_idx_p   Xmm(7)

#define JMP_TO_LABEL(label)                  \
    if (isa == cpu::x64::avx512_common) {    \
        kmovw(reg_tmp_32, k_mask);           \
    } else {                                 \
        uni_vmovmskps(reg_tmp_32, xmm_mask); \
    }                                        \
    and_(reg_tmp_32, 0x1);                   \
    cmp(reg_tmp_32, 0x0);                    \
    je(label, T_NEAR);

static inline bool isFloatCompatible(memory::data_type type) {
    return memory::data_type::f32 == type || memory::data_type::bf16 == type;
}

template <cpu_isa_t isa>
struct jit_uni_topk_kernel_f32 : public jit_uni_topk_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_topk_kernel_f32)

    explicit jit_uni_topk_kernel_f32(jit_topk_config_params jcp)
        : jit_uni_topk_kernel(jcp), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        load_emitter.reset(new jit_load_emitter(this, isa, nullptr));
        store_emitter.reset(new jit_store_emitter(this, isa, nullptr));

        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_dst_idx, ptr[reg_params + GET_OFF(index)]);
        mov(reg_prc, ptr[reg_params + GET_OFF(process)]);
        mov(reg_prc_idx, ptr[reg_params + GET_OFF(process_index)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

        mov(reg_table, l_table);

        data_type = MKLDNNExtensionUtils::IEPrecisionToDataType(jcp_.precision);

        if (jcp_.mode_max) {
            cmp_flg = _cmp_lt_os;       // if val[left] < val[right], set mask 1, swap
            heap_cmp_flg = _cmp_nle_us; // min heap is used for max topk, if a > b, set mask 1, swap
        } else {
            cmp_flg = _cmp_nle_us;      // if val[left] > val[right], set mask 1, swap
            heap_cmp_flg = _cmp_lt_os;  // max heap is used for min topk, if a < b, set mask 1, swap
        }

        if (isa == cpu::x64::avx512_common)
            uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        load_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx()), static_cast<size_t>(reg_load_table.getIdx())};
        store_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx())};
        store_pool_vec_idxs = {static_cast<size_t>(vmm_zero.getIdx())};

        topk_loop();

        this->postamble();

        load_emitter->emit_data();
        store_emitter->emit_data();

        prepare_idx_table();
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;
    size_t vlen = cpu_isa_traits<isa>::vlen;
    mkldnn::memory::data_type data_type;

    Xbyak::Address table_val(int index) { return ptr[reg_table + index * vlen]; }
    Xbyak::Address table_seq_val(int index) { return ptr[reg_table + jcp_.axis_dim * vlen + index * sizeof(float)]; }

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 reg_dst_idx = r10;
    Xbyak::Reg64 reg_prc = r11;
    Xbyak::Reg64 reg_prc_idx = r12;
    Xbyak::Reg64 reg_work_amount = r13;
    Xbyak::Reg64 reg_table = r14;
    Xbyak::Reg64 reg_params = abi_param1;
    Xbyak::Reg64 reg_j = rax;
    Xbyak::Reg64 reg_aux = rdx;
    Xbyak::Reg64 reg_aux_idx = rbx;

    Xbyak::Reg8 reg_tmp_8 = r15b;
    Xbyak::Reg32 reg_tmp_32 = r15d;
    Xbyak::Reg64 reg_tmp_64 = r15;

    Xbyak::Reg64 reg_load_table = rbp;
    Xbyak::Reg64 reg_load_store_mask = rsi;

    Xbyak::Reg64 reg_offset = reg_work_amount; // only for heap sort
    Xbyak::Reg64 reg_i = reg_load_table;       // only for heap sort
    Xbyak::Reg64 reg_k = reg_aux_idx;          // only for bubble sort

    Vmm vmm_zero = Vmm(0); // vmm_zero represents Vmm(0) when isa is avx512_common, otherwise vmm_mask represents Vmm(0)

    const Xbyak::Opmask k_mask = Xbyak::Opmask(1);
    const int step = vlen / sizeof(float);
    const int tail = jcp_.work_amount % step;
    const int topk_tail = jcp_.top_k % step;

    unsigned char cmp_flg;
    unsigned char heap_cmp_flg;

    Xbyak::Label l_table;

    std::unique_ptr<jit_load_emitter> load_emitter = nullptr;
    std::unique_ptr<jit_store_emitter> store_emitter = nullptr;

    std::vector<size_t> store_pool_gpr_idxs;
    std::vector<size_t> load_pool_gpr_idxs;
    std::vector<size_t> store_pool_vec_idxs;

    inline void topk_loop() {
        if (jcp_.algorithm == TopKAlgorithm::topk_bubble_sort) {
            if (jcp_.layout == TopKLayoutType::topk_blocked && jcp_.topk_innermost) {
                if (jcp_.top_k == 1) {
                    topk_bubble_horiz_blocked_innermost();
                } else {
                    topk_bubble_scalar_blocked_innermost();
                }
            } else {
                topk_bubble_vector();
            }
        } else if (jcp_.algorithm == TopKAlgorithm::topk_bitonic_sort) {
            if (jcp_.layout == TopKLayoutType::topk_blocked && jcp_.topk_innermost) {
                topk_bitonic_blocked_innermost();
            } else {
                topk_bitonic_vector();
            }
        } else if (jcp_.algorithm == TopKAlgorithm::topk_heap_sort) {
            topk_heap_scalar();
        }
    }

    inline void topk_bitonic_vector() {
        Xbyak::Label topk_main_loop_label;
        Xbyak::Label topk_main_loop_end_label;
        L(topk_main_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(topk_main_loop_end_label, T_NEAR);

            topk_bitonic(step);

            add(reg_src, step * jcp_.data_size);
            add(reg_dst, step * jcp_.data_size);
            add(reg_dst_idx, step * sizeof(int));
            sub(reg_work_amount, step);

            jmp(topk_main_loop_label, T_NEAR);
        }
        L(topk_main_loop_end_label);

        // tail
        if (tail) {
            Xbyak::Label topk_tail_loop_end_label;
            cmp(reg_work_amount, tail);
            jl(topk_tail_loop_end_label, T_NEAR);

            topk_bitonic(tail);

            L(topk_tail_loop_end_label);
        }
    }

    inline void topk_bitonic(int elt_num) {
        // src => prc
        for (int i = 0; i < jcp_.axis_dim; i++) {
            load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_tmp.getIdx())},
                          std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, elt_num, i * jcp_.sort_stride * jcp_.data_size),
                          {}, {load_pool_gpr_idxs});
            store_emitter->emit_code({static_cast<size_t>(vmm_tmp.getIdx())}, {static_cast<size_t>(reg_prc.getIdx())},
                           std::make_shared<store_emitter_context>(Precision::FP32, jcp_.precision, elt_num, i * jcp_.sort_stride * jcp_.data_size),
                           {store_pool_vec_idxs}, {store_pool_gpr_idxs});

            load_emitter->emit_code({static_cast<size_t>(reg_table.getIdx())}, {static_cast<size_t>(vmm_tmp.getIdx())},
                          std::make_shared<load_emitter_context>(Precision::I32, Precision::I32, elt_num, i * vlen),
                          {}, {load_pool_gpr_idxs});
            store_emitter->emit_code({static_cast<size_t>(vmm_tmp.getIdx())}, {static_cast<size_t>(reg_prc_idx.getIdx())},
                           std::make_shared<store_emitter_context>(Precision::I32, Precision::I32, elt_num, i * jcp_.sort_stride * sizeof(int)),
                           {store_pool_vec_idxs}, {store_pool_gpr_idxs});
        }

        // sort
        bitonic_sort_vector(elt_num);
        if (jcp_.sort_index) {
            bitonic_sort_vector(elt_num, false);
        }

        // prc => dst
        for (int i = 0; i < jcp_.top_k; i++) {
            load_emitter->emit_code({static_cast<size_t>(reg_prc.getIdx())}, {static_cast<size_t>(vmm_tmp.getIdx())},
                          std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, elt_num, i * jcp_.sort_stride * jcp_.data_size),
                          {}, {load_pool_gpr_idxs});
            store_emitter->emit_code({static_cast<size_t>(vmm_tmp.getIdx())}, {static_cast<size_t>(reg_dst.getIdx())},
                           std::make_shared<store_emitter_context>(Precision::FP32, jcp_.precision, elt_num, i * jcp_.sort_stride * jcp_.data_size),
                           {store_pool_vec_idxs}, {store_pool_gpr_idxs});

            load_emitter->emit_code({static_cast<size_t>(reg_prc_idx.getIdx())}, {static_cast<size_t>(vmm_tmp.getIdx())},
                          std::make_shared<load_emitter_context>(Precision::I32, Precision::I32, elt_num, i * jcp_.sort_stride * sizeof(int)),
                          {}, {load_pool_gpr_idxs});
            store_emitter->emit_code({static_cast<size_t>(vmm_tmp.getIdx())}, {static_cast<size_t>(reg_dst_idx.getIdx())},
                           std::make_shared<store_emitter_context>(Precision::I32, Precision::I32, elt_num, i * jcp_.sort_stride * sizeof(int)),
                           {store_pool_vec_idxs}, {store_pool_gpr_idxs});
        }
    }

    // src memory layout: (N) * (CB * H * W * blk_size)
    // prc memory layout: (C) * (N * H * W)
    // topk_bitonic_vector_blocked_innermost: sort (C) * (N * H * W / blk_size * blk_size) elements
    //                                        sort (C) * (N * H * W % blk_size) elements in the rear
    inline void topk_bitonic_blocked_innermost() {
        Xbyak::Label topk_main_loop_label;
        Xbyak::Label topk_main_loop_end_label;
        L(topk_main_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(topk_main_loop_end_label, T_NEAR);

            // src => prc
            blocked_innermost_load(step);

            // sort
            bitonic_sort_vector(step);
            if (jcp_.sort_index) {
                bitonic_sort_vector(step, false);
            }

            // prc => dst
            blocked_innermost_store(step);

            add(reg_src, step * jcp_.blk_size * jcp_.data_size);
            add(reg_dst, step * jcp_.blk_size * jcp_.data_size);
            add(reg_dst_idx, step * jcp_.blk_size * sizeof(int));
            sub(reg_work_amount, step);

            jmp(topk_main_loop_label, T_NEAR);
        }
        L(topk_main_loop_end_label);

        // tail exists because working buffer has planar layout, though source buffer has blocked layout)
        if (tail) {
            Xbyak::Label topk_tail_loop_end_label;
            cmp(reg_work_amount, tail);
            jl(topk_tail_loop_end_label, T_NEAR);

            // src => prc
            blocked_innermost_load(tail);

            bitonic_sort_vector(tail);
            if (jcp_.sort_index) {
                bitonic_sort_vector(tail, false);
            }

            // prc => dst
            blocked_innermost_store(tail);

            L(topk_tail_loop_end_label);
        }
    }

    inline void bitonic_sort_vector(int elt_num, bool cmp_val = true) {
        if (cmp_val) {
            mov(reg_j, jcp_.bitonic_idx_cnt);
            mov(reg_aux, ptr[reg_params + GET_OFF(bitonic_idx_buf)]);
        } else {
            mov(reg_j, jcp_.bitonic_k_idx_cnt);
            mov(reg_aux, ptr[reg_params + GET_OFF(bitonic_k_idx_buf)]);
        }

        Xbyak::Label topk_main_loop_label;
        Xbyak::Label topk_main_loop_end_label;
        L(topk_main_loop_label);
        {
            cmp(reg_j, 0);
            jle(topk_main_loop_end_label, T_NEAR);

            bitonic_swap_vector(elt_num, cmp_val);

            add(reg_aux, 2 * sizeof(int));
            sub(reg_j, 2);

            jmp(topk_main_loop_label, T_NEAR);
        }
        L(topk_main_loop_end_label);
    }

    inline void blocked_innermost_load(int elt_num) {
        for (int i = 0; i < jcp_.axis_dim; i++) {
            for (int j = 0; j < elt_num; j++) {
                int offset = i / jcp_.blk_size * jcp_.blk_stride + i % jcp_.blk_size + j * jcp_.blk_size;

                load_scalar(xmm_tmp, ptr[reg_src + offset * jcp_.data_size], data_type);
                store_scalar(ptr[reg_prc + (i * jcp_.sort_stride + j) * jcp_.data_size], xmm_tmp, data_type);

                uni_vmovdqu(xmm_tmp, table_val(i));
                store_scalar(ptr[reg_prc_idx + (i * jcp_.sort_stride + j) * sizeof(int)], xmm_tmp, memory::data_type::s32, false);
            }
        }
    }

    inline void blocked_innermost_store(int elt_num) {
        for (int i = 0; i < jcp_.top_k; i++) {
            for (int j = 0; j < elt_num; j++) {
                int offset = i / jcp_.blk_size * jcp_.blk_stride + i % jcp_.blk_size + j * jcp_.blk_size;

                load_scalar(xmm_tmp, ptr[reg_prc + (i * jcp_.sort_stride + j) * jcp_.data_size], data_type);
                store_scalar(ptr[reg_dst + offset * jcp_.data_size], xmm_tmp, data_type);

                load_scalar(xmm_tmp, ptr[reg_prc_idx + (i * jcp_.sort_stride + j) * sizeof(int)], memory::data_type::s32);
                store_scalar(ptr[reg_dst_idx + offset * sizeof(int)], xmm_tmp, memory::data_type::s32);
            }
        }
    }

    inline void bitonic_get_addr(Xbyak::Reg64 reg_base, int data_size, int offset = 0) {
        mov(reg_aux_idx.cvt32(), ptr[reg_aux + offset]);
        mul_by_const(reg_aux_idx, reg_tmp_64, data_size);
        add(reg_aux_idx, reg_base);
    }

    inline void bitonic_swap_vector(int elt_num, bool cmp_val = true) {
        bitonic_get_addr(reg_prc, jcp_.data_size, 0);
        load_emitter->emit_code({static_cast<size_t>(reg_aux_idx.getIdx())}, {static_cast<size_t>(vmm_val_l.getIdx())},
                      std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, elt_num),
                      {}, {load_pool_gpr_idxs});
        bitonic_get_addr(reg_prc, jcp_.data_size, sizeof(int));
        load_emitter->emit_code({static_cast<size_t>(reg_aux_idx.getIdx())}, {static_cast<size_t>(vmm_val_r.getIdx())},
                      std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, elt_num),
                      {}, {load_pool_gpr_idxs});
        bitonic_get_addr(reg_prc_idx, sizeof(int), 0);
        load_emitter->emit_code({static_cast<size_t>(reg_aux_idx.getIdx())}, {static_cast<size_t>(vmm_idx_l.getIdx())},
                      std::make_shared<load_emitter_context>(Precision::I32, Precision::FP32, elt_num),
                      {}, {load_pool_gpr_idxs});
        bitonic_get_addr(reg_prc_idx, sizeof(int), sizeof(int));
        load_emitter->emit_code({static_cast<size_t>(reg_aux_idx.getIdx())}, {static_cast<size_t>(vmm_idx_r.getIdx())},
                      std::make_shared<load_emitter_context>(Precision::I32, Precision::FP32, elt_num),
                      {}, {load_pool_gpr_idxs});

        swap_vector(vmm_val_l, vmm_idx_l, vmm_val_r, vmm_idx_r, cmp_val);

        bitonic_get_addr(reg_prc, jcp_.data_size, 0);
        store_emitter->emit_code({static_cast<size_t>(vmm_val_l.getIdx())}, {static_cast<size_t>(reg_aux_idx.getIdx())},
                       std::make_shared<store_emitter_context>(Precision::FP32, jcp_.precision, elt_num),
                       {store_pool_vec_idxs}, {store_pool_gpr_idxs});
        bitonic_get_addr(reg_prc, jcp_.data_size, sizeof(int));
        store_emitter->emit_code({static_cast<size_t>(vmm_val_r.getIdx())}, {static_cast<size_t>(reg_aux_idx.getIdx())},
                       std::make_shared<store_emitter_context>(Precision::FP32, jcp_.precision, elt_num),
                       {store_pool_vec_idxs}, {store_pool_gpr_idxs});
        bitonic_get_addr(reg_prc_idx, sizeof(int), 0);
        store_emitter->emit_code({static_cast<size_t>(vmm_idx_l.getIdx())}, {static_cast<size_t>(reg_aux_idx.getIdx())},
                       std::make_shared<store_emitter_context>(Precision::FP32, Precision::I32, elt_num),
                       {store_pool_vec_idxs}, {store_pool_gpr_idxs});
        bitonic_get_addr(reg_prc_idx, sizeof(int), sizeof(int));
        store_emitter->emit_code({static_cast<size_t>(vmm_idx_r.getIdx())}, {static_cast<size_t>(reg_aux_idx.getIdx())},
                       std::make_shared<store_emitter_context>(Precision::FP32, Precision::I32, elt_num),
                       {store_pool_vec_idxs}, {store_pool_gpr_idxs});
    }

    inline void topk_heap_scalar() {
        // init dst
        int i = 0;
        for (; i + step <= jcp_.top_k; i += step) {
            load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_tmp.getIdx())},
                                std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, step, i * jcp_.data_size),
                                {}, {load_pool_gpr_idxs});
            store_emitter->emit_code({static_cast<size_t>(vmm_tmp.getIdx())}, {static_cast<size_t>(reg_dst.getIdx())},
                           std::make_shared<store_emitter_context>(Precision::FP32, jcp_.precision, step, i * jcp_.data_size),
                           {store_pool_vec_idxs}, {store_pool_gpr_idxs});

            uni_vmovdqu(vmm_tmp, table_seq_val(i));
            store_emitter->emit_code({static_cast<size_t>(vmm_tmp.getIdx())}, {static_cast<size_t>(reg_dst_idx.getIdx())},
                           std::make_shared<store_emitter_context>(Precision::I32, Precision::I32, step, i * sizeof(int)),
                           {store_pool_vec_idxs}, {store_pool_gpr_idxs});
        }
        if (topk_tail) {
            load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_tmp.getIdx())},
                          std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, topk_tail, i * jcp_.data_size),
                          {}, {load_pool_gpr_idxs});
            store_emitter->emit_code({static_cast<size_t>(vmm_tmp.getIdx())}, {static_cast<size_t>(reg_dst.getIdx())},
                           std::make_shared<store_emitter_context>(Precision::FP32, jcp_.precision, topk_tail, i * jcp_.data_size),
                           {store_pool_vec_idxs}, {store_pool_gpr_idxs});

            load_emitter->emit_code({static_cast<size_t>(reg_table.getIdx())}, {static_cast<size_t>(vmm_tmp.getIdx())},
                          std::make_shared<load_emitter_context>(Precision::I32, Precision::I32, topk_tail, jcp_.axis_dim * vlen + i * sizeof(float)),
                          {}, {load_pool_gpr_idxs});
            store_emitter->emit_code({static_cast<size_t>(vmm_tmp.getIdx())}, {static_cast<size_t>(reg_dst_idx.getIdx())},
                           std::make_shared<store_emitter_context>(Precision::I32, Precision::I32, topk_tail, i * sizeof(int)),
                           {store_pool_vec_idxs}, {store_pool_gpr_idxs});
        }

        // heapify
        int end = (jcp_.top_k - 2) / 2;
        for (int i = end; i >= 0; i--) {
            heapipy_sub_tree(i, jcp_.top_k - 1);
        }

        // update
        Xbyak::Label topk_main_loop_label;
        Xbyak::Label topk_main_loop_end_label;
        mov(reg_i, jcp_.top_k);
        L(topk_main_loop_label);
        {
            cmp(reg_i, jcp_.axis_dim);
            jge(topk_main_loop_end_label, T_NEAR);

            Xbyak::Label topk_update_loop_end_label;
            mov(reg_aux, reg_i);
            mul_by_const(reg_aux, reg_tmp_64, jcp_.data_size);
            add(reg_aux, reg_src);
            load_scalar(xmm_val_p, ptr[reg_aux], data_type);

            mov(reg_aux, reg_i);
            mul_by_const(reg_aux, reg_tmp_64, vlen);
            add(reg_aux, reg_table);
            uni_vmovdqu(xmm_idx_p, ptr[reg_aux]);
            uni_vcvtdq2ps(xmm_idx_p, xmm_idx_p);
            load_scalar(xmm_val_l, ptr[reg_dst], data_type);
            load_scalar(xmm_idx_l, ptr[reg_dst_idx], memory::data_type::s32);

            heap_cmp_node(xmm_val_p, xmm_idx_p, xmm_val_l, xmm_idx_l);
            JMP_TO_LABEL(topk_update_loop_end_label);

            store_scalar(ptr[reg_dst], xmm_val_p, data_type);
            store_scalar(ptr[reg_dst_idx], xmm_idx_p, memory::data_type::s32);
            heapipy_sub_tree(0, jcp_.top_k - 1);

            L(topk_update_loop_end_label);

            add(reg_i, 1);
            jmp(topk_main_loop_label, T_NEAR);
        }
        L(topk_main_loop_end_label);

        // extract topk
        if (jcp_.sort_index) {
            // reheapify by index
            for (int i = end; i >= 0; i--) {
                heapipy_sub_tree(i, jcp_.top_k - 1, false);
            }

            // extract by index
            for (int i = jcp_.top_k - 1; i > 0; i--) {
                heap_swap_root(i);
                heapipy_sub_tree(0, i - 1, false);
            }
        } else {
            // extract by value
            for (int i = jcp_.top_k - 1; i > 0; i--) {
                heap_swap_root(i);
                heapipy_sub_tree(0, i - 1);
            }
        }
    }

    inline void heapipy_sub_tree(int i, int valid, bool cmp_val = true) {
        Xbyak::Label topk_heapify_loop_label;
        Xbyak::Label topk_heapify_loop_end_label;
        Xbyak::Label topk_lchild_loop_label;
        Xbyak::Label topk_rchild_loop_label;

        if (valid > 0) {
            int end = (valid - 1) / 2;
            mov(reg_j, i);
            mov(reg_aux, reg_dst);
            mov(reg_aux_idx, reg_dst_idx);
            add(reg_aux, i * jcp_.data_size);
            add(reg_aux_idx, i * sizeof(int));
            mov(reg_offset, (2 * i + 1) * jcp_.data_size);
            L(topk_heapify_loop_label);
            {
                cmp(reg_j, end);
                jg(topk_heapify_loop_end_label, T_NEAR);

                load_scalar(xmm_val_p, ptr[reg_aux], data_type);
                load_scalar(xmm_idx_p, ptr[reg_aux_idx], memory::data_type::s32);

                // compare lchild-rchild
                mov(reg_prc, reg_dst);
                add(reg_prc, reg_offset);
                mov(reg_prc_idx, reg_dst_idx);
                add(reg_prc_idx, reg_offset);
                load_scalar(xmm_val_l, ptr[reg_prc], data_type);
                load_scalar(xmm_idx_l, ptr[reg_prc_idx], memory::data_type::s32);
                add(reg_prc, jcp_.sort_stride * jcp_.data_size);
                add(reg_prc_idx, jcp_.sort_stride * jcp_.data_size);

                // if last valid parent has no rchild
                cmp(reg_j, valid / 2);
                jge(topk_lchild_loop_label, T_NEAR);

                load_scalar(xmm_val_r, ptr[reg_prc], data_type);
                load_scalar(xmm_idx_r, ptr[reg_prc_idx], memory::data_type::s32);

                heap_cmp_node(xmm_val_l, xmm_idx_l, xmm_val_r, xmm_idx_r, cmp_val);
                JMP_TO_LABEL(topk_lchild_loop_label);

                // compare node-rchild
                L(topk_rchild_loop_label);
                {
                    heap_cmp_node(xmm_val_p, xmm_idx_p, xmm_val_r, xmm_idx_r, cmp_val);
                    JMP_TO_LABEL(topk_heapify_loop_end_label);

                    heap_swap_node(xmm_val_p, xmm_idx_p, xmm_val_r, xmm_idx_r);
                    mov(reg_aux, reg_prc);
                    mov(reg_aux_idx, reg_prc_idx);
                    add(reg_offset, jcp_.sort_stride * jcp_.data_size);
                    shl(reg_offset, 1);
                    add(reg_offset, jcp_.sort_stride * jcp_.data_size);
                    shl(reg_j, 1);
                    add(reg_j, 2);
                    jmp(topk_heapify_loop_label, T_NEAR);
                }

                // compare node-lchild
                L(topk_lchild_loop_label);
                {
                    heap_cmp_node(xmm_val_p, xmm_idx_p, xmm_val_l, xmm_idx_l, cmp_val);
                    JMP_TO_LABEL(topk_heapify_loop_end_label);

                    sub(reg_prc, jcp_.sort_stride * jcp_.data_size);
                    sub(reg_prc_idx, jcp_.sort_stride * jcp_.data_size);
                    heap_swap_node(xmm_val_p, xmm_idx_p, xmm_val_l, xmm_idx_l);
                    mov(reg_aux, reg_prc);
                    mov(reg_aux_idx, reg_prc_idx);
                    shl(reg_offset, 1);
                    add(reg_offset, jcp_.sort_stride * jcp_.data_size);
                    shl(reg_j, 1);
                    add(reg_j, 1);
                    jmp(topk_heapify_loop_label, T_NEAR);
                }
            }
            L(topk_heapify_loop_end_label);
        }
    }

    inline void heap_cmp_node(Xmm xmm_val_a, Xmm xmm_idx_a, Xmm xmm_val_b, Xmm xmm_idx_b, bool cmp_val = true) {
        if (isa == cpu::x64::avx512_common) {
            if (cmp_val)
                vcmpps(k_mask, xmm_val_a, xmm_val_b, heap_cmp_flg);
            else
                vcmpps(k_mask, xmm_idx_a, xmm_idx_b, _cmp_lt_os);
        } else if (isa == cpu::x64::avx2) {
            if (cmp_val)
                vcmpps(xmm_mask, xmm_val_a, xmm_val_b, heap_cmp_flg);
            else
                vcmpps(xmm_mask, xmm_idx_a, xmm_idx_b, _cmp_lt_os);
        } else {
            if (cmp_val) {
                movups(xmm_mask, xmm_val_a);
                cmpps(xmm_mask, xmm_val_b, heap_cmp_flg);
            } else {
                movups(xmm_mask, xmm_idx_a);
                cmpps(xmm_mask, xmm_idx_b, _cmp_lt_os);
            }
        }
    }

    // n: node, c: child
    inline void heap_swap_node(Xmm xmm_val_n, Xmm xmm_idx_n, Xmm xmm_val_c, Xmm xmm_idx_c) {
        // swap store
        store_scalar(ptr[reg_aux], xmm_val_c, data_type);
        store_scalar(ptr[reg_aux_idx], xmm_idx_c, memory::data_type::s32);
        store_scalar(ptr[reg_prc], xmm_val_n, data_type);
        store_scalar(ptr[reg_prc_idx], xmm_idx_n, memory::data_type::s32);
    }

    inline void heap_swap_root(int i) {
        load_scalar(xmm_val_p, ptr[reg_dst], data_type);
        load_scalar(xmm_idx_p, ptr[reg_dst_idx], memory::data_type::s32);
        load_scalar(xmm_val_l, ptr[reg_dst + i * jcp_.data_size], data_type);
        load_scalar(xmm_idx_l, ptr[reg_dst_idx + i * sizeof(int)], memory::data_type::s32);
        store_scalar(ptr[reg_dst], xmm_val_l, data_type);
        store_scalar(ptr[reg_dst_idx], xmm_idx_l, memory::data_type::s32);
        store_scalar(ptr[reg_dst + i * jcp_.data_size], xmm_val_p, data_type);
        store_scalar(ptr[reg_dst_idx + i * sizeof(int)], xmm_idx_p, memory::data_type::s32);
    }

    inline void topk_bubble_vector() {
        Xbyak::Label topk_main_loop_label;
        Xbyak::Label topk_main_loop_end_label;
        L(topk_main_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(topk_main_loop_end_label, T_NEAR);

            if (jcp_.bubble_inplace) {
                topk_bubble_inplace(step);
            } else {
                topk_bubble(step);
            }

            add(reg_src, step * jcp_.data_size);
            add(reg_dst, step * jcp_.data_size);
            add(reg_dst_idx, step * sizeof(int));
            sub(reg_work_amount, step);

            jmp(topk_main_loop_label, T_NEAR);
        }
        L(topk_main_loop_end_label);

        // tail
        if (tail) {
            Xbyak::Label topk_tail_loop_end_label;
            cmp(reg_work_amount, tail);
            jl(topk_tail_loop_end_label, T_NEAR);

            if (jcp_.bubble_inplace) {
                topk_bubble_inplace(tail);
            } else {
                topk_bubble(tail);
            }

            L(topk_tail_loop_end_label);
        }
    }

    inline void topk_bubble(int elt_num) {
        // init dst
        for (int i = 0; i < jcp_.top_k; i++) {
            load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_tmp.getIdx())},
                          std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, elt_num, i * jcp_.sort_stride * jcp_.data_size),
                          {}, {load_pool_gpr_idxs});
            store_emitter->emit_code({static_cast<size_t>(vmm_tmp.getIdx())}, {static_cast<size_t>(reg_dst.getIdx())},
                       std::make_shared<store_emitter_context>(Precision::FP32, jcp_.precision, elt_num, i * jcp_.sort_stride * jcp_.data_size),
                       {store_pool_vec_idxs}, {store_pool_gpr_idxs});

            uni_vmovdqu(vmm_tmp, table_val(i));
            store_emitter->emit_code({static_cast<size_t>(vmm_tmp.getIdx())}, {static_cast<size_t>(reg_dst_idx.getIdx())},
                       std::make_shared<store_emitter_context>(Precision::I32, Precision::I32, elt_num, i * jcp_.sort_stride * sizeof(int)),
                       {store_pool_vec_idxs}, {store_pool_gpr_idxs});
        }
        // sort
        for (int i = 0; i < jcp_.top_k - 1; i++) {
            for (int j = jcp_.top_k - 1; j > i; j--) {
                bubble_swap_vector(j - 1, j, elt_num);
            }
        }
        // update
        Xbyak::Label topk_update_loop_label;
        Xbyak::Label topk_update_loop_end_label;
        mov(reg_k, jcp_.top_k);
        L(topk_update_loop_label);
        {
            cmp(reg_k, jcp_.axis_dim);
            jge(topk_update_loop_end_label, T_NEAR);

            mov(reg_aux, reg_k);
            mul_by_const(reg_aux, reg_tmp_64, jcp_.sort_stride * jcp_.data_size);
            add(reg_aux, reg_src);
            load_emitter->emit_code({static_cast<size_t>(reg_aux.getIdx())}, {static_cast<size_t>(vmm_val_r.getIdx())},
                          std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, elt_num),
                          {}, {load_pool_gpr_idxs});

            mov(reg_aux, reg_k);
            mul_by_const(reg_aux, reg_tmp_64, vlen);
            add(reg_aux, reg_table);
            uni_vmovdqu(vmm_idx_r, ptr[reg_aux]);
            uni_vcvtdq2ps(vmm_idx_r, vmm_idx_r);
            for (int j = jcp_.top_k; j > 0; j--) {
                bubble_swap_vector(j - 1, j, elt_num);
            }

            add(reg_k, 1);
            jmp(topk_update_loop_label, T_NEAR);
        }
        L(topk_update_loop_end_label);

        if (jcp_.sort_index) {
            for (int i = 0; i < jcp_.top_k - 1; i++) {
                for (int j = jcp_.top_k - 1; j > i; j--) {
                    bubble_swap_vector(j - 1, j, elt_num, false);
                }
            }
        }
    }

    inline void topk_bubble_inplace(int elt_num) {
        // load
        for (int i = 0; i < jcp_.top_k; i++) {
            load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val(i).getIdx())},
                          std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, elt_num, i * jcp_.sort_stride * jcp_.data_size),
                          {}, {load_pool_gpr_idxs});
            uni_vmovdqu(vmm_idx(i), table_val(i));
            uni_vcvtdq2ps(vmm_idx(i), vmm_idx(i));
        }
        // sort
        for (int i = 0; i < jcp_.top_k - 1; i++) {
            for (int j = jcp_.top_k - 1; j > i; j--) {
                swap_vector(vmm_val(j - 1), vmm_idx(j - 1), vmm_val(j), vmm_idx(j));
            }
        }
        for (int i = jcp_.top_k; i < jcp_.axis_dim; i++) {
            load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val(jcp_.top_k).getIdx())},
                          std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, elt_num, i * jcp_.sort_stride * jcp_.data_size),
                          {}, {load_pool_gpr_idxs});
            uni_vmovdqu(vmm_idx(jcp_.top_k), table_val(i));
            uni_vcvtdq2ps(vmm_idx(jcp_.top_k), vmm_idx(jcp_.top_k));
            for (int j = jcp_.top_k; j > 0; j--) {
                swap_vector(vmm_val(j - 1), vmm_idx(j - 1), vmm_val(j), vmm_idx(j));
            }
        }
        if (jcp_.sort_index) {
            for (int i = 0; i < jcp_.top_k - 1; i++) {
                for (int j = jcp_.top_k - 1; j > i; j--) {
                    swap_vector(vmm_val(j - 1), vmm_idx(j - 1), vmm_val(j), vmm_idx(j), false);
                }
            }
        }
        // store
        for (int i = 0; i < jcp_.top_k; i++) {
            store_emitter->emit_code({static_cast<size_t>(vmm_val(i).getIdx())}, {static_cast<size_t>(reg_dst.getIdx())},
                       std::make_shared<store_emitter_context>(Precision::FP32, jcp_.precision, elt_num, i * jcp_.sort_stride * jcp_.data_size),
                       {store_pool_vec_idxs}, {store_pool_gpr_idxs});
            store_emitter->emit_code({static_cast<size_t>(vmm_idx(i).getIdx())}, {static_cast<size_t>(reg_dst_idx.getIdx())},
                       std::make_shared<store_emitter_context>(Precision::FP32, Precision::I32, elt_num, i * jcp_.sort_stride * sizeof(int)),
                       {store_pool_vec_idxs}, {store_pool_gpr_idxs});
        }
    }

    inline void topk_bubble_horiz_blocked_innermost() {
        // load and sort
        int i = 0;
        if (jcp_.axis_dim < jcp_.blk_size) {
            load_scalar(xmm_val(0), ptr[reg_src], data_type);
            uni_vmovdqu(xmm_idx(0), table_val(0));
            uni_vcvtdq2ps(xmm_idx(0), xmm_idx(0));
            i = 1;
        } else {
            load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val(0).getIdx())},
                          std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, step, 0),
                          {}, {load_pool_gpr_idxs});
            uni_vmovdqu(vmm_idx(0), table_seq_val(0));
            uni_vcvtdq2ps(vmm_idx(0), vmm_idx(0));
            if (isa == cpu::x64::sse41) {
                load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val(1).getIdx())},
                              std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, step, 4 * jcp_.data_size),
                              {}, {load_pool_gpr_idxs});
                uni_vmovdqu(vmm_idx(1), table_seq_val(4));
                uni_vcvtdq2ps(vmm_idx(1), vmm_idx(1));
                swap_vector(vmm_val(0), vmm_idx(0), vmm_val(1), vmm_idx(1));
            }
            i = jcp_.blk_size;
            for (; i + jcp_.blk_size <= jcp_.axis_dim; i += jcp_.blk_size) {
                int offset = i / jcp_.blk_size * jcp_.blk_stride;
                load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val(1).getIdx())},
                              std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, step, offset * jcp_.data_size),
                              {}, {load_pool_gpr_idxs});
                uni_vmovdqu(vmm_idx(1), table_seq_val(i));
                uni_vcvtdq2ps(vmm_idx(1), vmm_idx(1));
                swap_vector(vmm_val(0), vmm_idx(0), vmm_val(1), vmm_idx(1));
                if (isa == cpu::x64::sse41) {
                    load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val(1).getIdx())},
                                  std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, step, (offset + 4) * jcp_.data_size),
                                  {}, {load_pool_gpr_idxs});
                    uni_vmovdqu(vmm_idx(1), table_seq_val(i + 4));
                    uni_vcvtdq2ps(vmm_idx(1), vmm_idx(1));
                    swap_vector(vmm_val(0), vmm_idx(0), vmm_val(1), vmm_idx(1));
                }
            }
            horiz_process();
        }
        for (; i < jcp_.axis_dim; i++) {
            int offset = i / jcp_.blk_size * jcp_.blk_stride + i % jcp_.blk_size;
            load_scalar(xmm_val(1), ptr[reg_src + offset * jcp_.data_size], data_type);
            uni_vmovdqu(xmm_idx(1), table_val(i));
            uni_vcvtdq2ps(xmm_idx(1), xmm_idx(1));
            swap_scalar(xmm_val(0), xmm_idx(0), xmm_val(1), xmm_idx(1));
        }
        // store
        store_scalar(ptr[reg_dst], xmm_val(0), data_type);
        store_scalar(ptr[reg_dst_idx], xmm_idx(0), memory::data_type::s32);
    }

    // dst: xmm_val(0) and xmm_idx(0)
    // aux: xmm_val(2/3/4) and xmm_idx(2/3/4)
    inline void horiz_process() {
        if (isa == cpu::x64::sse41) {
            horize_top1();
        } else if (isa == cpu::x64::avx2) {
            Xbyak::Ymm ymm_val_dst = Xbyak::Ymm(vmm_val(0).getIdx());
            vextractf128(xmm_val(2), ymm_val_dst, 0);
            vextractf128(xmm_val(3), ymm_val_dst, 1);
            Xbyak::Ymm ymm_idx_dst = Xbyak::Ymm(vmm_idx(0).getIdx());
            vextractf128(xmm_idx(2), ymm_idx_dst, 0);
            vextractf128(xmm_idx(3), ymm_idx_dst, 1);
            swap_scalar(xmm_val(2), xmm_idx(2), xmm_val(3), xmm_idx(3));
            movups(xmm_val(0), xmm_val(2));
            movups(xmm_idx(0), xmm_idx(2));
            horize_top1();
        } else {
            Xbyak::Zmm zmm_val_dst = Xbyak::Zmm(vmm_val(0).getIdx());
            vextractf32x4(xmm_val(2), zmm_val_dst, 0);
            vextractf32x4(xmm_val(3), zmm_val_dst, 1);
            Xbyak::Zmm zmm_idx_dst = Xbyak::Zmm(vmm_idx(0).getIdx());
            vextractf32x4(xmm_idx(2), zmm_idx_dst, 0);
            vextractf32x4(xmm_idx(3), zmm_idx_dst, 1);
            swap_scalar(xmm_val(2), xmm_idx(2), xmm_val(3), xmm_idx(3));
            vextractf32x4(xmm_val(3), zmm_val_dst, 2);
            vextractf32x4(xmm_val(4), zmm_val_dst, 3);
            vextractf32x4(xmm_idx(3), zmm_idx_dst, 2);
            vextractf32x4(xmm_idx(4), zmm_idx_dst, 3);
            swap_scalar(xmm_val(3), xmm_idx(3), xmm_val(4), xmm_idx(4));
            swap_scalar(xmm_val(2), xmm_idx(2), xmm_val(3), xmm_idx(3));
            movups(xmm_val(0), xmm_val(2));
            movups(xmm_idx(0), xmm_idx(2));
            horize_top1();
        }
    }

    // dst: xmm_val(0) and xmm_idx(0)
    // aux: xmm_val(3) and xmm_idx(3)
    inline void horize_top1() {
        movshdup(xmm_val(3), xmm_val(0));                            // dst:1,2,3,4; aux:2,2,4,4
        movshdup(xmm_idx(3), xmm_idx(0));
        swap_scalar(xmm_val(0), xmm_idx(0), xmm_val(3), xmm_idx(3)); // dst:f(1,2),f(2,2),f(3,4),f(4,4)
        movhlps(xmm_val(3), xmm_val(0));                             // aux:f(3,4),f(4,4),4,4
        movhlps(xmm_idx(3), xmm_idx(0));
        swap_scalar(xmm_val(0), xmm_idx(0), xmm_val(3), xmm_idx(3)); // dst:f(1,2,3,4),...
    }

    inline void topk_bubble_scalar_blocked_innermost() {
        if (jcp_.bubble_inplace) {
            topk_bubble_scalar_inplace();
        } else {
            topk_bubble_scalar();
        }
    }

    inline void topk_bubble_scalar() {
        // init dst
        for (int i = 0; i < jcp_.top_k; i++) {
            int offset = i / jcp_.blk_size * jcp_.blk_stride + i % jcp_.blk_size;
            load_scalar(xmm_tmp, ptr[reg_src + offset * jcp_.data_size], data_type);
            store_scalar(ptr[reg_dst + offset * jcp_.data_size], xmm_tmp, data_type);

            uni_vmovdqu(xmm_tmp, table_val(i));
            store_scalar(ptr[reg_dst_idx + offset * sizeof(int)], xmm_tmp, memory::data_type::s32, false);
        }
        // sort
        for (int i = 0; i < jcp_.top_k - 1; i++) {
            for (int j = jcp_.top_k - 1; j > i; j--) {
                bubble_swap_scalar(j - 1, j);
            }
        }
        // update
        for (int i = jcp_.top_k; i < jcp_.axis_dim; i++) {
            int offset = i / jcp_.blk_size * jcp_.blk_stride + i % jcp_.blk_size;
            load_scalar(xmm_val_r, ptr[reg_src + offset * jcp_.data_size], data_type);
            uni_vmovdqu(xmm_idx_r, table_val(i));
            uni_vcvtdq2ps(xmm_idx_r, xmm_idx_r);
            for (int j = jcp_.top_k; j > 0; j--) {
                bubble_swap_scalar(j - 1, j);
            }
        }
        if (jcp_.sort_index) {
            for (int i = 0; i < jcp_.top_k - 1; i++) {
                for (int j = jcp_.top_k - 1; j > i; j--) {
                    bubble_swap_scalar(j - 1, j, false);
                }
            }
        }
    }

    inline void topk_bubble_scalar_inplace() {
        // load
        for (int i = 0; i < jcp_.top_k; i++) {
            int offset = i / jcp_.blk_size * jcp_.blk_stride + i % jcp_.blk_size;
            load_scalar(xmm_val(i), ptr[reg_src + offset * jcp_.data_size], data_type);
            uni_vmovdqu(xmm_idx(i), table_val(i));
            uni_vcvtdq2ps(xmm_idx(i), xmm_idx(i));
        }
        // sort
        for (int i = 0; i < jcp_.top_k - 1; i++) {
            for (int j = jcp_.top_k - 1; j > i; j--) {
                swap_scalar(xmm_val(j - 1), xmm_idx(j - 1), xmm_val(j), xmm_idx(j));
            }
        }
        for (int i = jcp_.top_k; i < jcp_.axis_dim; i++) {
            int offset = i / jcp_.blk_size * jcp_.blk_stride + i % jcp_.blk_size;
            load_scalar(xmm_val(jcp_.top_k), ptr[reg_src + offset * jcp_.data_size], data_type);
            uni_vmovdqu(xmm_idx(jcp_.top_k), table_val(i));
            uni_vcvtdq2ps(xmm_idx(jcp_.top_k), xmm_idx(jcp_.top_k));
            for (int j = jcp_.top_k; j > 0; j--) {
                swap_scalar(xmm_val(j - 1), xmm_idx(j - 1), xmm_val(j), xmm_idx(j));
            }
        }
        if (jcp_.sort_index) {
            for (int i = 0; i < jcp_.top_k - 1; i++) {
                for (int j = jcp_.top_k - 1; j > i; j--) {
                    swap_scalar(xmm_val(j - 1), xmm_idx(j - 1), xmm_val(j), xmm_idx(j), false);
                }
            }
        }
        // store
        for (int i = 0; i < jcp_.top_k; i++) {
            int offset = i / jcp_.blk_size * jcp_.blk_stride + i % jcp_.blk_size;
            store_scalar(ptr[reg_dst + offset * jcp_.data_size], xmm_val(i), data_type);
            store_scalar(ptr[reg_dst_idx + offset * sizeof(int)], xmm_idx(i), memory::data_type::s32);
        }
    }

    inline void bubble_swap_vector(int l, int r, int elt_num, bool cmp_val = true) {
        load_emitter->emit_code({static_cast<size_t>(reg_dst.getIdx())}, {static_cast<size_t>(vmm_val_l.getIdx())},
                      std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, elt_num, l * jcp_.sort_stride * jcp_.data_size),
                      {}, {load_pool_gpr_idxs});
        load_emitter->emit_code({static_cast<size_t>(reg_dst_idx.getIdx())}, {static_cast<size_t>(vmm_idx_l.getIdx())},
                      std::make_shared<load_emitter_context>(Precision::I32, Precision::FP32, elt_num, l * jcp_.sort_stride * sizeof(int)),
                      {}, {load_pool_gpr_idxs});
        if (r != jcp_.top_k) {
            load_emitter->emit_code({static_cast<size_t>(reg_dst.getIdx())}, {static_cast<size_t>(vmm_val_r.getIdx())},
                          std::make_shared<load_emitter_context>(jcp_.precision, Precision::FP32, elt_num, r * jcp_.sort_stride * jcp_.data_size),
                          {}, {load_pool_gpr_idxs});
            load_emitter->emit_code({static_cast<size_t>(reg_dst_idx.getIdx())}, {static_cast<size_t>(vmm_idx_r.getIdx())},
                          std::make_shared<load_emitter_context>(Precision::I32, Precision::FP32, elt_num, r * jcp_.sort_stride * sizeof(int)),
                          {}, {load_pool_gpr_idxs});
        }

        swap_vector(vmm_val_l, vmm_idx_l, vmm_val_r, vmm_idx_r, cmp_val);

        store_emitter->emit_code({static_cast<size_t>(vmm_val_l.getIdx())}, {static_cast<size_t>(reg_dst.getIdx())},
                       std::make_shared<store_emitter_context>(Precision::FP32, jcp_.precision, elt_num, l * jcp_.sort_stride * jcp_.data_size),
                       {store_pool_vec_idxs}, {store_pool_gpr_idxs});
        store_emitter->emit_code({static_cast<size_t>(vmm_idx_l.getIdx())}, {static_cast<size_t>(reg_dst_idx.getIdx())},
                       std::make_shared<store_emitter_context>(Precision::FP32, Precision::I32, elt_num, l * jcp_.sort_stride * sizeof(int)),
                       {store_pool_vec_idxs}, {store_pool_gpr_idxs});
        if (r != jcp_.top_k) {
            store_emitter->emit_code({static_cast<size_t>(vmm_val_r.getIdx())}, {static_cast<size_t>(reg_dst.getIdx())},
                           std::make_shared<store_emitter_context>(Precision::FP32, jcp_.precision, elt_num, r * jcp_.sort_stride * jcp_.data_size),
                           {store_pool_vec_idxs}, {store_pool_gpr_idxs});
            store_emitter->emit_code({static_cast<size_t>(vmm_idx_r.getIdx())}, {static_cast<size_t>(reg_dst_idx.getIdx())},
                           std::make_shared<store_emitter_context>(Precision::FP32, Precision::I32, elt_num, r * jcp_.sort_stride * sizeof(int)),
                           {store_pool_vec_idxs}, {store_pool_gpr_idxs});
        }
    }

    inline void swap_vector(Vmm vmm_val_a, Vmm vmm_idx_a, Vmm vmm_val_b, Vmm vmm_idx_b, bool cmp_val = true) {
        if (isa == cpu::x64::avx512_common) {
            if (cmp_val)
                vcmpps(k_mask, vmm_val_a, vmm_val_b, cmp_flg);
            else
                vcmpps(k_mask, vmm_idx_a, vmm_idx_b, _cmp_nle_us);

            uni_vmovups(vmm_tmp, vmm_val_a);
            vblendmps(vmm_val_a | k_mask, vmm_val_a, vmm_val_b);
            vblendmps(vmm_val_b | k_mask, vmm_val_b, vmm_tmp);

            uni_vmovups(vmm_tmp, vmm_idx_a);
            vblendmps(vmm_idx_a | k_mask, vmm_idx_a, vmm_idx_b);
            vblendmps(vmm_idx_b | k_mask, vmm_idx_b, vmm_tmp);
        } else if (isa == cpu::x64::avx2) {
            if (cmp_val)
                vcmpps(vmm_mask, vmm_val_a, vmm_val_b, cmp_flg);
            else
                vcmpps(vmm_mask, vmm_idx_a, vmm_idx_b, _cmp_nle_us);

            uni_vmovups(vmm_tmp, vmm_val_a);
            vblendvps(vmm_val_a, vmm_val_a, vmm_val_b, vmm_mask);
            vblendvps(vmm_val_b, vmm_val_b, vmm_tmp, vmm_mask);

            uni_vmovups(vmm_tmp, vmm_idx_a);
            vblendvps(vmm_idx_a, vmm_idx_a, vmm_idx_b, vmm_mask);
            vblendvps(vmm_idx_b, vmm_idx_b, vmm_tmp, vmm_mask);
        } else {
            if (cmp_val) {
                movups(vmm_mask, vmm_val_a);
                cmpps(vmm_mask, vmm_val_b, cmp_flg);
            } else {
                movups(vmm_mask, vmm_idx_a);
                cmpps(vmm_mask, vmm_idx_b, _cmp_nle_us);
            }

            uni_vmovups(vmm_tmp, vmm_val_a);
            blendvps(vmm_val_a, vmm_val_b);
            blendvps(vmm_val_b, vmm_tmp);

            uni_vmovups(vmm_tmp, vmm_idx_a);
            blendvps(vmm_idx_a, vmm_idx_b);
            blendvps(vmm_idx_b, vmm_tmp);
        }
    }

    inline void bubble_swap_scalar(int l, int r, bool cmp_val = true) {
        int offset_l = l / jcp_.blk_size * jcp_.blk_stride + l % jcp_.blk_size;
        int offset_r = r / jcp_.blk_size * jcp_.blk_stride + r % jcp_.blk_size;

        load_scalar(xmm_val_l, ptr[reg_dst + offset_l * jcp_.data_size], data_type);
        load_scalar(xmm_idx_l, ptr[reg_dst_idx + offset_l * sizeof(int)], memory::data_type::s32);
        if (r != jcp_.top_k) {
            load_scalar(xmm_val_r, ptr[reg_dst + offset_r * jcp_.data_size], data_type);
            load_scalar(xmm_idx_r, ptr[reg_dst_idx + offset_r * sizeof(int)], memory::data_type::s32);
        }

        swap_scalar(xmm_val_l, xmm_idx_l, xmm_val_r, xmm_idx_r, cmp_val);

        store_scalar(ptr[reg_dst + offset_l * jcp_.data_size], xmm_val_l, data_type);
        store_scalar(ptr[reg_dst_idx + offset_l * sizeof(int)], xmm_idx_l, memory::data_type::s32);
        if (r != jcp_.top_k) {
            store_scalar(ptr[reg_dst + offset_r * jcp_.data_size], xmm_val_r, data_type);
            store_scalar(ptr[reg_dst_idx + offset_r * sizeof(int)], xmm_idx_r, memory::data_type::s32);
        }
    }

    inline void swap_scalar(Xmm xmm_val_a, Xmm xmm_idx_a, Xmm xmm_val_b, Xmm xmm_idx_b, bool cmp_val = true) {
        if (isa == cpu::x64::avx512_common) {
            if (cmp_val)
                vcmpps(k_mask, xmm_val_a, xmm_val_b, cmp_flg);
            else
                vcmpps(k_mask, xmm_idx_a, xmm_idx_b, _cmp_nle_us);

            uni_vmovups(xmm_tmp, xmm_val_a);
            vblendmps(xmm_val_a | k_mask, xmm_val_a, xmm_val_b);
            vblendmps(xmm_val_b | k_mask, xmm_val_b, xmm_tmp);

            uni_vmovups(xmm_tmp, xmm_idx_a);
            vblendmps(xmm_idx_a | k_mask, xmm_idx_a, xmm_idx_b);
            vblendmps(xmm_idx_b | k_mask, xmm_idx_b, xmm_tmp);
        } else if (isa == cpu::x64::avx2) {
            if (cmp_val)
                vcmpps(xmm_mask, xmm_val_a, xmm_val_b, cmp_flg);
            else
                vcmpps(xmm_mask, xmm_idx_a, xmm_idx_b, _cmp_nle_us);

            uni_vmovups(xmm_tmp, xmm_val_a);
            vblendvps(xmm_val_a, xmm_val_a, xmm_val_b, xmm_mask);
            vblendvps(xmm_val_b, xmm_val_b, xmm_tmp, xmm_mask);

            uni_vmovups(xmm_tmp, xmm_idx_a);
            vblendvps(xmm_idx_a, xmm_idx_a, xmm_idx_b, xmm_mask);
            vblendvps(xmm_idx_b, xmm_idx_b, xmm_tmp, xmm_mask);
        } else {
            if (cmp_val) {
                movups(xmm_mask, xmm_val_a);
                cmpps(xmm_mask, xmm_val_b, cmp_flg);
            } else {
                movups(xmm_mask, xmm_idx_a);
                cmpps(xmm_mask, xmm_idx_b, _cmp_nle_us);
            }

            uni_vmovups(xmm_tmp, xmm_val_a);
            blendvps(xmm_val_a, xmm_val_b);
            blendvps(xmm_val_b, xmm_tmp);

            uni_vmovups(xmm_tmp, xmm_idx_a);
            blendvps(xmm_idx_a, xmm_idx_b);
            blendvps(xmm_idx_b, xmm_tmp);
        }
    }

    inline void load_scalar(Xmm xmm_src, const Xbyak::Address &op, memory::data_type src_dt, bool cvt_dt = true) {
        switch (src_dt) {
            case memory::data_type::f32:
            case memory::data_type::s32:
                movss(xmm_src, op);
                break;
            case memory::data_type::bf16:
                pinsrw(xmm_src, op, 0x0);
                uni_vpslld(xmm_src, xmm_src, 16);
                break;
            case memory::data_type::s8:
                movsx(reg_tmp_32, op);
                movq(xmm_src, reg_tmp_64);
                break;
            case memory::data_type::u8:
                movzx(reg_tmp_32, op);
                movq(xmm_src, reg_tmp_64);
                break;
            default:
                assert(!"unknown src_dt");
        }

        if (cvt_dt && !isFloatCompatible(src_dt)) {
            uni_vcvtdq2ps(xmm_src, xmm_src);
        }
    }

    inline void store_scalar(const Xbyak::Address &op, Xmm xmm_dst, memory::data_type dst_dt, bool cvt_dt = true) {
        if (cvt_dt && !isFloatCompatible(dst_dt)) {
            uni_vcvtps2dq(xmm_dst, xmm_dst);
        }

        switch (dst_dt) {
            case memory::data_type::f32:
            case memory::data_type::s32:
                movss(op, xmm_dst);
                break;
            case memory::data_type::bf16:
                uni_vpsrld(xmm_dst, xmm_dst, 16);
                pextrw(op, xmm_dst, 0x0);
                break;
            case memory::data_type::s8:
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            case memory::data_type::u8:
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            default:
                assert(!"unknown dst_dt");
        }
    }

    void prepare_idx_table() {
        auto broadcast_int = [&](int val) {
            for (size_t d = 0; d < vlen / sizeof(float); ++d) {
                dd(val);
            }
        };

        align(64);
        L(l_table);

        // 00000000 11111111 22222222 ...
        for (int i = 0; i < jcp_.axis_dim; i++) {
            broadcast_int(i);
        }

        // 01234567 89...
        for (int i = 0; i < jcp_.axis_dim; i++) {
            dd(i);
        }
    }
};

bool MKLDNNTopKNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (isDynamicNgraphNode(op)) {
            errorMessage = "Doesn't support op with dynamic shapes";
            return false;
        }
        const auto topKOp = ngraph::as_type_ptr<const ngraph::op::v1::TopK>(op);
        if (!topKOp) {
            errorMessage = "Node is not an instance of the TopK from the operations set v1 or v3";
            return false;
        }
        auto topKConst = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(topKOp->get_input_node_shared_ptr(TOPK_K));
        if (!topKConst) {
            errorMessage = "Second tensor is not constant";
            return false;
        }

        if (topKOp->get_mode() != ngraph::op::TopKMode::MAX &&
                topKOp->get_mode() != ngraph::op::TopKMode::MIN) {
            errorMessage = "Unsupported mode.";
            return false;
        }
        if (!one_of(topKOp->get_sort_type(), ngraph::op::TopKSortType::NONE,
                                  ngraph::op::TopKSortType::SORT_VALUES,
                                  ngraph::op::TopKSortType::SORT_INDICES)) {
            errorMessage = "Unsupported sort type.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNTopKNode::MKLDNNTopKNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "TopK layer with name '" + getName() + "'";

        auto topKOp = ngraph::as_type_ptr<ngraph::op::v1::TopK>(op);

        src_dims = topKOp->get_input_shape(TOPK_DATA);
        dst_dims = topKOp->get_output_shape(TOPK_DATA);
        dst_idx_dims = topKOp->get_output_shape(TOPK_INDEX);
        src_dims_size = src_dims.size();
        dst_dims_size = dst_dims.size();

        top_k = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(topKOp->get_input_node_shared_ptr(TOPK_K))->cast_vector<int>()[0];

        axis = topKOp->get_axis();

        if (topKOp->get_mode() == ngraph::op::TopKMode::MAX)
            mode_max = true;
        else
            mode_max = false;

        if (topKOp->get_sort_type() == ngraph::op::TopKSortType::SORT_INDICES)
            sort_index = true;
        else
            sort_index = false;
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void MKLDNNTopKNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != 2 || getChildEdges().size() < 2)
        IE_THROW() << errorPrefix << " gets incorrect number of input/output edges!";

    if (getInputShapeAtPort(TOPK_DATA).getRank() != getOutputShapeAtPort(TOPK_DATA).getRank()) {
        IE_THROW() << errorPrefix << " gets incorrect number of input/output dimensions!";
    }
    if (getInputShapeAtPort(TOPK_K).getRank() != 1) {
        IE_THROW() << errorPrefix << " gets incorrect index vector dimension! Index vector should be 1 dimension.";
    }

    if (dst_dims != dst_idx_dims)
        IE_THROW() << errorPrefix << " gets incorrect output tensor dimension sizes!";

    if (axis < 0)
        axis += src_dims_size;
    if (axis < 0 || axis >= static_cast<int>(src_dims_size))
        IE_THROW() << errorPrefix << " gets incorrect input parameters dimensions and axis number!";
    axis_dim = src_dims[axis];

    if (top_k > src_dims[axis])
        IE_THROW() << errorPrefix << " gets top_k out of range!";
}

void MKLDNNTopKNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    impl_desc_type impl_type;
    if (mayiuse(cpu::x64::avx512_common)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (mayiuse(cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    } else if (mayiuse(cpu::x64::sse41)) {
        impl_type = impl_desc_type::jit_sse42;
    } else {
        impl_type = impl_desc_type::ref;
    }

    jit_mode = mayiuse(cpu::x64::sse41);

    static const Precision supportedPrecision[] = {
        Precision::FP32,
        Precision::BF16,
        Precision::I32,
        Precision::I8,
        Precision::U8
    };

    Precision dataPrecision = getOriginalOutputPrecisionAtPort(TOPK_DATA);
    if (dataPrecision == Precision::BF16 && !mayiuse(avx512_core))
        IE_THROW() << errorPrefix << " gets incorrect isa for BF16! AVX512 must be supported!";
    bool precisionSupported = std::find(std::begin(supportedPrecision), std::end(supportedPrecision), dataPrecision)
                                     != std::end(supportedPrecision);
    if (!precisionSupported) {
        if (dataPrecision.is_float()) {
            dataPrecision = Precision::FP32;
        } else {
            dataPrecision = Precision::I32;
        }
    }

    std::vector<std::pair<LayoutType, LayoutType>> dataFomats{
        {LayoutType::ncsp, LayoutType::ncsp},
        {LayoutType::nspc, LayoutType::nspc},
        {LayoutType::nCsp16c, LayoutType::nCsp16c},
        {LayoutType::nCsp8c, LayoutType::nCsp8c}
    };

    for (const auto &df : dataFomats) {
        addSupportedPrimDesc({{df.first, dataPrecision}, {LayoutType::ncsp, Precision::I32}},
                             {{df.second, dataPrecision}, {df.second, Precision::I32}},
                             impl_type);
    }
}

void MKLDNNTopKNode::createPrimitive() {
    auto &dstMemPtr = getChildEdgeAt(TOPK_DATA)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(TOPK_DATA)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        IE_THROW() << errorPrefix << " has not allocated destination memory.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        IE_THROW() << errorPrefix << " has not allocate input memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << errorPrefix << " has nullable preferable primitive descriptor";

    if (getParentEdgeAt(0)->getMemory().getDesc().hasLayoutType(LayoutType::ncsp)) {
        layout = TopKLayoutType::topk_ncsp;
    } else if (getParentEdgeAt(0)->getMemory().getDesc().hasLayoutType(LayoutType::nspc)) {
        layout = TopKLayoutType::topk_nspc;
    } else {
        layout = TopKLayoutType::topk_blocked;
    }

    auto selectedPD = getSelectedPrimitiveDescriptor();
    auto data_type = MKLDNNExtensionUtils::IEPrecisionToDataType(selectedPD->getConfig().inConfs[TOPK_DATA].desc->getPrecision());
    data_size = MKLDNNExtensionUtils::sizeOfDataType(data_type);

    topk_innermost = false;
    if ((layout == TopKLayoutType::topk_ncsp && axis == dst_dims_size - 1) ||
       ((layout == TopKLayoutType::topk_nspc || layout == TopKLayoutType::topk_blocked) && axis == 1))
        topk_innermost = true;

    if (mayiuse(cpu::x64::avx512_common)) {
        blk_size = 16;
        count_xmm = 16; // only 16 vector registers are valid in sse instructions
    } else if (mayiuse(cpu::x64::sse41)) {
        blk_size = 8;
        count_xmm = 16;
    }

    // [case 1]: if 2 * (top_k + 1) + 2 <= count_xmm, thus top_k is small enough that the vector registers are sufficient
    //           to keep all necessary data for sorting, no need to load and store frequently, use inplace bubble sort;
    // [case 2]: only when topk is imposed on innermost dimsension of planar(ncsp/nspc) layout, should heap sort be used;
    // [case 3]: by default, use bitonic sort when alg_cost_bitonic < alg_cost_bubble, otherwise use bubble sort.
    //           alg_cost_bitonic = (N / 4) * logN * (logN + 1)
    //           alg_cost_bubble = K * (K - 1) / 2 + (N - K) * K
    //           where, N = axis_dim, K = topk_k
    //           the above two alg_costs are not the exact implementation costs, yet it's proper to use them to decide
    //           which algorithm should be used for specific N and K.
    if (top_k <= count_xmm / 2 - 2) {
        algorithm = TopKAlgorithm::topk_bubble_sort;
        bubble_inplace = true;
    } else if ((layout == TopKLayoutType::topk_ncsp || layout == TopKLayoutType::topk_nspc) && topk_innermost) {
        algorithm = TopKAlgorithm::topk_heap_sort;
    } else {
        auto log_axis_dim = log2(axis_dim);
        size_t alg_cost_bitonic = static_cast<size_t>((axis_dim / 4.0f) * log_axis_dim * (log_axis_dim + 1));
        size_t alg_cost_bubble = top_k * (top_k - 1) / 2 + (axis_dim - top_k) * top_k;
        if (alg_cost_bitonic < alg_cost_bubble) {
            algorithm = TopKAlgorithm::topk_bitonic_sort;
        } else {
            algorithm = TopKAlgorithm::topk_bubble_sort;
            bubble_inplace = false;
        }
    }

    if (jit_mode) {
        auto layout_dims = getChildEdgeAt(TOPK_DATA)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getBlockDims();
        calc_dims_size(layout_dims);

        auto jcp = jit_topk_config_params();
        jcp.precision = selectedPD->getConfig().inConfs[TOPK_DATA].desc->getPrecision();
        jcp.data_size = data_size;
        jcp.blk_size = blk_size;
        jcp.layout = layout;
        jcp.top_k = top_k;
        jcp.axis_dim = axis_dim;
        jcp.mode_max = mode_max;
        jcp.sort_index = sort_index;
        jcp.topk_innermost = topk_innermost;
        jcp.algorithm = algorithm;
        jcp.bubble_inplace = bubble_inplace;
        jcp.sort_stride = static_cast<int>(I);
        jcp.work_amount = static_cast<int>(I);
        if (layout == TopKLayoutType::topk_blocked && topk_innermost) {
            jcp.blk_stride = I * blk_size;
            if (algorithm == TopKAlgorithm::topk_bubble_sort) {
                jcp.work_amount = static_cast<int>(axis_dim);
            }
        }

        if (algorithm == TopKAlgorithm::topk_bitonic_sort) {
            size_t src_count = srcMemPtr->GetDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();
            vec_process_ptr.resize(src_count * data_size);
            vec_process_idx_ptr.resize(src_count * sizeof(int32_t));

            calc_bitonic_idx(axis_dim, jcp.bitonic_size, jcp.bitonic_idx_cnt, true);
            if (sort_index) {
                calc_bitonic_idx(top_k, jcp.bitonic_k_size, jcp.bitonic_k_idx_cnt, false);
            }
        }

        if (mayiuse(cpu::x64::avx512_common)) {
            topk_kernel.reset(new jit_uni_topk_kernel_f32<cpu::x64::avx512_common>(jcp));
        } else if (mayiuse(cpu::x64::avx2)) {
            topk_kernel.reset(new jit_uni_topk_kernel_f32<cpu::x64::avx2>(jcp));
        } else if (mayiuse(cpu::x64::sse41)) {
            topk_kernel.reset(new jit_uni_topk_kernel_f32<cpu::x64::sse41>(jcp));
        }

        if (topk_kernel)
            topk_kernel->create_ker();
    } else { //reference mode
        int j;
        for (j = src_dims.size() - 1; j >= 0; j--) {
            if (src_dims[j] != 1)
                break;
        }
        if (static_cast<size_t>(j) == axis)
            is_last_dim = true;
        dim = static_cast<int>(src_dims[axis]);
        before_num = count(src_dims, 0, axis);
    }
}

void MKLDNNTopKNode::execute(mkldnn::stream strm) {
    auto &srcMemPtr = getParentEdgeAt(TOPK_DATA)->getMemoryPtr();
    auto &dstMemPtr = getChildEdgeAt(TOPK_DATA)->getMemoryPtr();
    auto &dstIndexesMemPtr = getChildEdgeAt(TOPK_INDEX)->getMemoryPtr();

    const uint8_t *src_data = reinterpret_cast<const uint8_t *>(srcMemPtr->GetPtr());
    uint8_t *dst_data = reinterpret_cast<uint8_t *>(dstMemPtr->GetPtr());
    uint8_t *dst_idx = reinterpret_cast<uint8_t *>(dstIndexesMemPtr->GetPtr());

    if (jit_mode) {
        topk_process(src_data, dst_data, dst_idx);
    } else {
        if (layout == TopKLayoutType::topk_ncsp) {
            auto in_ptr = reinterpret_cast<const float *>(src_data);
            auto out_ptr = reinterpret_cast<float *>(dst_data);
            auto out_idx_ptr = reinterpret_cast<int32_t *>(dst_idx);
            topk_ref(in_ptr, out_ptr, out_idx_ptr);
        } else {
            IE_THROW() << errorPrefix <<  "only support plain layout on machine w/o sse42.";
        }
    }
}

void MKLDNNTopKNode::topk_process(const uint8_t *in_ptr, uint8_t *out_ptr, uint8_t *out_idx_ptr) {
    uint8_t *process_ptr = vec_process_ptr.data();
    uint8_t *process_idx_ptr = vec_process_idx_ptr.data();

    // [blocked layout with topk on C]
    if (layout == TopKLayoutType::topk_blocked && topk_innermost) {
        size_t IA = div_up(src_dims[1], blk_size);
        size_t OA = div_up(dst_dims[1], blk_size);
        if (algorithm == TopKAlgorithm::topk_bubble_sort) {
            parallel_for2d(O, I, [&](size_t o, size_t i) {
                const uint8_t *in_ptr_a = in_ptr + (o * IA * I + i) * blk_size * data_size;
                uint8_t *out_ptr_a = out_ptr + (o * OA * I + i) * blk_size * data_size;
                uint8_t *out_idx_ptr_a = out_idx_ptr + (o * OA * I + i) * blk_size * sizeof(int32_t);
                size_t work_amount = 1;
                topk_kernel_process(in_ptr_a, out_ptr_a, out_idx_ptr_a, NULL, NULL, work_amount);
            });
        } else if (algorithm == TopKAlgorithm::topk_bitonic_sort) {
            parallel_for(O, [&](size_t o) {
                const uint8_t *in_ptr_a = in_ptr + o * IA * I * blk_size * data_size;
                uint8_t *process_ptr_a = process_ptr + o * IA * I * blk_size * data_size;
                uint8_t *process_idx_ptr_a = process_idx_ptr + o * IA * I * blk_size * sizeof(int32_t);
                uint8_t *out_ptr_a = out_ptr + o * OA * I * blk_size * data_size;
                uint8_t *out_idx_ptr_a = out_idx_ptr + o * OA * I * blk_size * sizeof(int32_t);
                size_t work_amount = I;
                topk_kernel_process(in_ptr_a, out_ptr_a, out_idx_ptr_a, process_ptr_a, process_idx_ptr_a, work_amount);
            });
        }
    } else { // [planar layout] [blocked layout with topk on non-C]
        parallel_for2d(O, I / blk_size, [&](size_t o, size_t k) {
            const uint8_t *in_ptr_a = in_ptr + (o * A * I + k * blk_size) * data_size;
            uint8_t *process_ptr_a = process_ptr + (o * A * I + k * blk_size) * data_size;
            uint8_t *process_idx_ptr_a = process_idx_ptr + (o * A * I + k * blk_size) * sizeof(int32_t);
            uint8_t *out_ptr_a = out_ptr + (o * top_k * I + k * blk_size) * data_size;
            uint8_t *out_idx_ptr_a = out_idx_ptr + (o * top_k * I + k * blk_size) * sizeof(int32_t);
            size_t work_amount = blk_size;
            topk_kernel_process(in_ptr_a, out_ptr_a, out_idx_ptr_a, process_ptr_a, process_idx_ptr_a, work_amount);
        });

        size_t tail_start = I / blk_size * blk_size;
        size_t work_amount = I - tail_start;
        if (work_amount) {
            parallel_for(O, [&](size_t o) {
                const uint8_t *in_ptr_a = in_ptr + (o * A * I + tail_start) * data_size;
                uint8_t *process_ptr_a = process_ptr + (o * A * I + tail_start) * data_size;
                uint8_t *process_idx_ptr_a = process_idx_ptr + (o * A * I + tail_start) * sizeof(int32_t);
                uint8_t *out_ptr_a = out_ptr + (o * top_k * I + tail_start) * data_size;
                uint8_t *out_idx_ptr_a = out_idx_ptr + (o * top_k * I + tail_start) * sizeof(int32_t);
                topk_kernel_process(in_ptr_a, out_ptr_a, out_idx_ptr_a, process_ptr_a, process_idx_ptr_a, work_amount);
            });
        }
    }
}

inline void MKLDNNTopKNode::topk_kernel_process(const uint8_t *in_p, uint8_t *out_p, uint8_t *out_idx_p,
                                                uint8_t *process_p, uint8_t *process_idx_p, size_t work_amount) {
    auto arg = jit_topk_call_args();
    arg.src = static_cast<const void *>(in_p);
    arg.process = static_cast<void *>(process_p);
    arg.process_index = static_cast<void *>(process_idx_p);
    arg.dst = static_cast<void *>(out_p);
    arg.index = static_cast<void *>(out_idx_p);
    arg.work_amount = work_amount;
    arg.bitonic_idx_buf = vec_bitonic_idx.data();
    arg.bitonic_k_idx_buf = vec_bitonic_k_idx.data();
    (*topk_kernel)(&arg);
}

// bitonic_size: length of the total array for sorting, power of 2
//          len: length of the array being sorted
//        start: start index of the array being sorted
//      sub_len: half of len
//    sub_start: start index of the sub array being sorted
//    minor_len: half of sub_len
//            n: number of valid elements in bitonic sort
//            p: pow of 2 number, so that p/2 < n <= p
//   empty tail: p-n elements in the rear don't need sorting,
inline void MKLDNNTopKNode::bitonic_push_idx(int p, int n, std::vector<int> &vec, int &cnt, bool cmp_val) {
    // memory stride of adjacent elements in sorting
    int sort_stride = static_cast<int>(I);
    cnt = 0;
    for (int len = 2; len < p; len <<= 1) {
        for (int start = 0; start < p; start += len) {
            int sub_len = len >> 1;
            // empty tail
            for (int i = sub_len - 1; start + len - i - 1 < n && i >= 0; i--) {
                vec[cnt++] = (start + i) * sort_stride;
                vec[cnt++] = (start + len - i - 1) * sort_stride;
            }
            for (; sub_len > 0; sub_len >>= 1) {
                for (int sub_start = start; sub_start < start + len; sub_start += sub_len) {
                    int minor_len = sub_len >> 1;
                    // empty tail
                    for (int j = 0; sub_start + j + minor_len < n && j < minor_len; j++) {
                        vec[cnt++] = (sub_start + j) * sort_stride;
                        vec[cnt++] = (sub_start + j + minor_len) * sort_stride;
                    }
                }
            }
        }
    }

    // last round sort
    int sub_p = p >> 1;
    for (int i = sub_p - 1; p - i - 1 < n && i >= 0; i--) {
        vec[cnt++] = i * sort_stride;
        vec[cnt++] = (p - i - 1) * sort_stride;
    }
    for (; sub_p > 0; sub_p >>= 1) {
        // support partial sort as well as full sort
        for (int sub_start = 0; (!cmp_val || (cmp_val && sub_start < n)) && sub_start < p;
             sub_start += sub_p) {
            int minor_p = sub_p >> 1;
            for (int j = 0; sub_start + j + minor_p < n && j < minor_p; j++) {
                vec[cnt++] = (sub_start + j) * sort_stride;
                vec[cnt++] = (sub_start + j + minor_p) * sort_stride;
            }
        }
    }
}

void MKLDNNTopKNode::calc_bitonic_idx(size_t n, int &p, int &cnt, bool cmp_val) {
    int m = n - 1;
    int log_p = 0;
    p = 1;
    while (m) {
        p <<= 1;
        m >>= 1;
        log_p++;
    }

    // maximum times of bitonic comparison: (p / 4) * log_p * (log_p + 1)
    // each comparison need two indices
    int max_cnt = (p >> 1) * log_p * (log_p + 1);
    if (cmp_val) {
        vec_bitonic_idx.resize(max_cnt);
        bitonic_push_idx(p, n, vec_bitonic_idx, cnt, cmp_val);
    } else {
        vec_bitonic_k_idx.resize(max_cnt);
        bitonic_push_idx(p, n, vec_bitonic_k_idx, cnt, cmp_val);
    }
}

// O: total size of the outer dimensions
// A: size of the topk imposed dimension
// I: total size of the inner dimensions
void MKLDNNTopKNode::calc_dims_size(const SizeVector &layout_dims) {
    O = 1, I = 1;
    A = src_dims[axis];
    int layout_axis = axis;
    if (layout == TopKLayoutType::topk_nspc) {
        layout_axis = axis == 0 ? 0 : (axis == 1 ? static_cast<int>(layout_dims.size() - 1) : axis - 1);
    }

    for (int i = 0; i < layout_axis; i++)
        O *= layout_dims[i];
    for (size_t i = layout_axis + 1; i < layout_dims.size(); i++)
        I *= layout_dims[i];
    if (layout == TopKLayoutType::topk_blocked && topk_innermost) {
        I /= blk_size;
    }
}

void MKLDNNTopKNode::topk_ref(const float *in_ptr, float *out_ptr, int32_t *dst_idx) {
    if (mode_max)
        topk_ref_process(in_ptr, out_ptr, dst_idx, src_dims, [](float x, float y)->float { return x > y; });
    else
        topk_ref_process(in_ptr, out_ptr, dst_idx, src_dims, [](float x, float y)->float { return x < y; });
}

void MKLDNNTopKNode::topk_ref_process(const float* src_data, float* dst_data, int32_t* dst_idx, const SizeVector &in_dims,
                               std::function<float(float, float)> compare) const {
    int after_num = count(in_dims, axis + 1, in_dims.size());

    parallel_for2d(before_num, after_num, [&](int i0, int i1) {
        std::vector<float> max_values(top_k + 1);
        std::vector<int> max_indexes(top_k + 1);
        float tmp_value;
        int tmp_index;
        int s_index = i0 * dim * after_num + i1;

        auto swap_func = [&](int index1, int index2) {
            tmp_value = max_values[index1];
            max_values[index1] = max_values[index2];
            max_values[index2] = tmp_value;

            tmp_index = max_indexes[index1];
            max_indexes[index1] = max_indexes[index2];
            max_indexes[index2] = tmp_index;
        };

        for (int i2 = 0; i2 < top_k; i2++) {
            max_values[i2] = src_data[s_index];
            max_indexes[i2] = i2;
            s_index += after_num;
        }
        for (int i2 = 0; i2 < top_k - 1; i2++) {
            for (int i3 = top_k - 1; i3 > i2; i3--) {
                if (compare(max_values[i3], max_values[i3 - 1])) {
                    swap_func(i3, i3 - 1);
                }
            }
        }
        for (int i2 = top_k; i2 < dim; i2++) {
            max_values[top_k] = src_data[s_index];
            max_indexes[top_k] = i2;
            for (int i3 = top_k; i3 > 0; i3--) {
                if (compare(max_values[i3], max_values[i3 - 1]))
                    swap_func(i3, i3 - 1);
                else
                    break;
            }
            s_index += after_num;
        }
        if (sort_index) {
            for (int i2 = 0; i2 < top_k - 1; i2++) {
                for (int i3 = top_k - 1; i3 > i2; i3--) {
                    if (std::greater<int>()(max_indexes[i3 - 1], max_indexes[i3])) {
                        swap_func(i3, i3 - 1);
                    }
                }
            }
        }
        if (dst_data) {
            for (int i2 = 0; i2 < top_k; i2++)
                dst_data[i0 * top_k * after_num + i2 * after_num + i1] = max_values[i2];
        }
        if (dst_idx) {
            for (int i2 = 0; i2 < top_k; i2++)
                dst_idx[i0 * top_k * after_num + i2 * after_num + i1] = max_indexes[i2];
        }
    });
}

inline int MKLDNNTopKNode::count(SizeVector dims, size_t start_ind, size_t end_ind) {
    size_t count = 1;
    for (size_t i = start_ind; i < end_ind; i++)
        count *= dims[i];
    return static_cast<int>(count);
}

inline int MKLDNNTopKNode::count(SizeVector dims, size_t start_ind) {
    return count(dims, start_ind, dims.size());
}

bool MKLDNNTopKNode::created() const {
    return getType() == TopK;
}

REG_MKLDNN_PRIM_FOR(MKLDNNTopKNode, TopK);
