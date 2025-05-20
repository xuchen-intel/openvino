// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "topk_kernel.hpp"

#include "cpu/x64/jit_uni_eltwise.hpp"
#include "dnnl_extension_utils.h"
#include "emitters/plugin/x64/jit_load_store_emitters.hpp"
#include "onednn/dnnl.h"

using namespace dnnl;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {
namespace kernel {

#define GET_OFF(field) offsetof(jit_topk_call_args, field)

#define vmm_mask   Vmm(0)
#define vmm_tmp    Vmm(1)
#define vmm_val(i) Vmm(2 * (i) + 2)
#define vmm_idx(i) Vmm(2 * (i) + 3)
#define vmm_val_l  Vmm(2)
#define vmm_idx_l  Vmm(3)
#define vmm_val_r  Vmm(4)
#define vmm_idx_r  Vmm(5)

#define xmm_mask   Xmm(0)
#define xmm_tmp    Xmm(1)
#define xmm_val(i) Xmm(2 * (i) + 2)
#define xmm_idx(i) Xmm(2 * (i) + 3)
#define xmm_val_l  Xmm(2)
#define xmm_idx_l  Xmm(3)
#define xmm_val_r  Xmm(4)
#define xmm_idx_r  Xmm(5)

#define xmm_val_p Xmm(6)
#define xmm_idx_p Xmm(7)

#define JMP_TO_LABEL(label)                  \
    if (isa == cpu::x64::avx512_core) {      \
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
void jit_uni_topk_kernel_f32<isa>::create_ker() {
    jit_generator::create_kernel();
    ker_ = (decltype(ker_))jit_ker();
}

template <cpu_isa_t isa>
void jit_uni_topk_kernel_f32<isa>::generate() {
    this->preamble();

    mov(reg_src, ptr[reg_params + GET_OFF(src)]);
    mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
    mov(reg_dst_idx, ptr[reg_params + GET_OFF(index)]);
    mov(reg_prc, ptr[reg_params + GET_OFF(process)]);
    mov(reg_prc_idx, ptr[reg_params + GET_OFF(process_index)]);
    mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

    bool shape_agnostic_alg = jcp_.algorithm == TopKAlgorithm::topk_heap_sort ||
                                (jcp_.algorithm == TopKAlgorithm::topk_bubble_sort && !jcp_.bubble_inplace);

    if (!shape_agnostic_alg) {
        mov(reg_table, l_table);
    }

    data_type = DnnlExtensionUtils::ElementTypeToDataType(jcp_.precision);
    precision_in_reg = isFloatCompatible(data_type) ? ov::element::f32 : ov::element::i32;
    if (!shape_agnostic_alg && jcp_.layout == TopKLayoutType::topk_blocked && jcp_.topk_innermost) {
        blk_stride = jcp_.sort_stride * jcp_.blk_size;
    }

    if (jcp_.mode_max) {
        cmp_flg = _cmp_lt_os;        // if val[left] < val[right], set mask 1, swap
        heap_cmp_flg = _cmp_nle_us;  // min heap is used for max topk, if a > b, set mask 1, swap
    } else {
        cmp_flg = _cmp_nle_us;      // if val[left] > val[right], set mask 1, swap
        heap_cmp_flg = _cmp_lt_os;  // max heap is used for min topk, if a < b, set mask 1, swap
    }

    if (isa == cpu::x64::avx512_core) {
        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
    }

    load_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx()),
                            static_cast<size_t>(reg_load_table.getIdx())};
    store_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx())};
    store_pool_vec_idxs = {static_cast<size_t>(vmm_zero.getIdx())};

    topk_loop();

    this->postamble();

    emit_emitters_data();

    if (!shape_agnostic_alg) {
        prepare_idx_table();
    }
}

template <cpu_isa_t isa>
void jit_uni_topk_kernel_f32<isa>::emit_emitters_data() {
    for (const auto& emitter : emitters) {
        emitter.second->emit_data();
    }
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::load(Xbyak::Reg64 reg_src, Vmm vmm_src, const int elt_num, const int offset = 0) {
    emit_load(reg_src, vmm_src, jcp_.precision, precision_in_reg, elt_num, offset);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::load_i32(Xbyak::Reg64 reg_src, Vmm vmm_src, const int elt_num, const int offset = 0) {
    emit_load(reg_src, vmm_src, ov::element::i32, ov::element::i32, elt_num, offset);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::store(Vmm vmm_dst, Xbyak::Reg64 reg_dst, const int elt_num, const int offset = 0) {
    emit_store(vmm_dst, reg_dst, precision_in_reg, jcp_.precision, elt_num, offset);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::store_i32(Vmm vmm_dst, Xbyak::Reg64 reg_dst, const int elt_num, const int offset = 0) {
    emit_store(vmm_dst, reg_dst, ov::element::i32, ov::element::i32, elt_num, offset);
}
template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::emit_load(Xbyak::Reg64 reg_src,
                        Vmm vmm_src,
                        ov::element::Type src_prc,
                        ov::element::Type dst_prc,
                        const int elt_num,
                        const int offset = 0) {
    const auto seed = load_emitter_params(src_prc, dst_prc, elt_num).hash();
    if (!emitters[seed]) {
        emitters[seed] = std::make_unique<jit_load_emitter>(this, isa, src_prc, dst_prc, elt_num);
    }

    emitters[seed]->emit_code({static_cast<size_t>(reg_src.getIdx()), static_cast<size_t>(offset)},
                                {static_cast<size_t>(vmm_src.getIdx())},
                                {},
                                {load_pool_gpr_idxs});
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::emit_store(Vmm vmm_dst,
                        Xbyak::Reg64 reg_dst,
                        ov::element::Type src_prc,
                        ov::element::Type dst_prc,
                        const int elt_num,
                        const int offset = 0) {
    const auto seed = store_emitter_params(src_prc, dst_prc, elt_num).hash();
    if (!emitters[seed]) {
        emitters[seed] = std::make_unique<jit_store_emitter>(this, isa, src_prc, dst_prc, elt_num);
    }

    // for cases when Store emitter need 2 aux vmm we can use vmm_dst as second aux vmm
    std::vector<size_t> local_store_pool_vec_idxs = {static_cast<size_t>(vmm_dst.getIdx())};
    local_store_pool_vec_idxs.insert(local_store_pool_vec_idxs.begin(),
                                        store_pool_vec_idxs.begin(),
                                        store_pool_vec_idxs.end());

    emitters[seed]->emit_code({static_cast<size_t>(vmm_dst.getIdx())},
                                {static_cast<size_t>(reg_dst.getIdx()), static_cast<size_t>(offset)},
                                {local_store_pool_vec_idxs},
                                {store_pool_gpr_idxs});
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::topk_loop() {
    if (jcp_.algorithm == TopKAlgorithm::topk_bubble_sort) {
        if (jcp_.layout == TopKLayoutType::topk_blocked && jcp_.topk_innermost) {
            if (jcp_.top_k == 1 && !jcp_.stable) {
                topk_bubble_horiz();
            } else {
                topk_bubble_BLK_on_channel_verti();
            }
        } else if (jcp_.topk_innermost && jcp_.top_k == 1 && !jcp_.stable) {
            topk_bubble_horiz();
        } else {
            topk_bubble_vector();
        }
    } else if (jcp_.algorithm == TopKAlgorithm::topk_bitonic_sort) {
        if (jcp_.layout == TopKLayoutType::topk_blocked && jcp_.topk_innermost) {
            topk_bitonic_BLK_on_channel();
        } else {
            topk_bitonic_vector();
        }
    } else if (jcp_.algorithm == TopKAlgorithm::topk_heap_sort) {
        topk_heap_sorting();
    }
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::topk_bitonic_vector() {
    Xbyak::Label topk_main_loop_label;
    Xbyak::Label topk_main_loop_end_label;
    L(topk_main_loop_label);
    {
        cmp(reg_work_amount, vector_step);
        jl(topk_main_loop_end_label, T_NEAR);

        topk_bitonic(vector_step);

        add(reg_src, vector_step * jcp_.data_size);
        add(reg_dst, vector_step * jcp_.data_size);
        add(reg_dst_idx, vector_step * sizeof(int));
        sub(reg_work_amount, vector_step);

        jmp(topk_main_loop_label, T_NEAR);
    }
    L(topk_main_loop_end_label);

    // tail
    if (tail_step) {
        Xbyak::Label topk_tail_loop_end_label;
        cmp(reg_work_amount, tail_step);
        jl(topk_tail_loop_end_label, T_NEAR);

        topk_bitonic(tail_step);

        L(topk_tail_loop_end_label);
    }
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::topk_bitonic(int elt_num) {
    // src => prc
    for (int i = 0; i < jcp_.axis_dim; i++) {
        load(reg_src, vmm_tmp, elt_num, i * jcp_.sort_stride * jcp_.data_size);
        store(vmm_tmp, reg_prc, elt_num, i * jcp_.sort_stride * jcp_.data_size);

        load_i32(reg_table, vmm_tmp, elt_num, i * vlen);
        store_i32(vmm_tmp, reg_prc_idx, elt_num, i * jcp_.sort_stride * sizeof(int));
    }

    // sort
    bitonic_sort_vector(elt_num);
    if (jcp_.sort_index) {
        bitonic_sort_vector(elt_num, false);
    }

    // prc => dst
    for (int i = 0; i < jcp_.top_k; i++) {
        load(reg_prc, vmm_tmp, elt_num, i * jcp_.sort_stride * jcp_.data_size);
        store(vmm_tmp, reg_dst, elt_num, i * jcp_.sort_stride * jcp_.data_size);

        load_i32(reg_prc_idx, vmm_tmp, elt_num, i * jcp_.sort_stride * sizeof(int));
        store_i32(vmm_tmp, reg_dst_idx, elt_num, i * jcp_.sort_stride * sizeof(int));
    }
}

// src memory layout: (N) * (CB * H * W * blk_size)
// prc memory layout: (C) * (N * H * W)
// topk_bitonic_BLK_on_channel: sort (C) * (N * H * W / blk_size * blk_size) elements
//                              sort (C) * (N * H * W % blk_size) elements in the rear
template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::topk_bitonic_BLK_on_channel() {
    Xbyak::Label topk_main_loop_label;
    Xbyak::Label topk_main_loop_end_label;
    L(topk_main_loop_label);
    {
        cmp(reg_work_amount, vector_step);
        jl(topk_main_loop_end_label, T_NEAR);

        // src => prc
        bitonic_BLK_on_channel_load(vector_step);

        // sort
        bitonic_sort_vector(vector_step);
        if (jcp_.sort_index) {
            bitonic_sort_vector(vector_step, false);
        }

        // prc => dst
        bitonic_BLK_on_channel_store(vector_step);

        add(reg_src, vector_step * jcp_.blk_size * jcp_.data_size);
        add(reg_dst, vector_step * jcp_.blk_size * jcp_.data_size);
        add(reg_dst_idx, vector_step * jcp_.blk_size * sizeof(int));
        sub(reg_work_amount, vector_step);

        jmp(topk_main_loop_label, T_NEAR);
    }
    L(topk_main_loop_end_label);

    // tail exists because working buffer has planar layout, though source buffer has blocked layout)
    if (tail_step) {
        Xbyak::Label topk_tail_loop_end_label;
        cmp(reg_work_amount, tail_step);
        jl(topk_tail_loop_end_label, T_NEAR);

        // src => prc
        bitonic_BLK_on_channel_load(tail_step);

        bitonic_sort_vector(tail_step);
        if (jcp_.sort_index) {
            bitonic_sort_vector(tail_step, false);
        }

        // prc => dst
        bitonic_BLK_on_channel_store(tail_step);

        L(topk_tail_loop_end_label);
    }
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::bitonic_sort_vector(int elt_num, bool cmp_val = true) {
    if (cmp_val) {
        mov(reg_i, jcp_.bitonic_idx_cnt);
        mov(reg_aux, ptr[reg_params + GET_OFF(bitonic_idx_buf)]);
    } else {
        mov(reg_i, jcp_.bitonic_k_idx_cnt);
        mov(reg_aux, ptr[reg_params + GET_OFF(bitonic_k_idx_buf)]);
    }

    Xbyak::Label topk_main_loop_label;
    Xbyak::Label topk_main_loop_end_label;
    L(topk_main_loop_label);
    {
        cmp(reg_i, 0);
        je(topk_main_loop_end_label, T_NEAR);

        bitonic_swap_vector(elt_num, cmp_val);

        add(reg_aux, 2 * sizeof(int));
        sub(reg_i, 2);

        jmp(topk_main_loop_label, T_NEAR);
    }
    L(topk_main_loop_end_label);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::bitonic_BLK_on_channel_load(int elt_num) {
    for (int i = 0; i < jcp_.axis_dim; i++) {
        for (int j = 0; j < elt_num; j++) {
            int offset = i / jcp_.blk_size * blk_stride + i % jcp_.blk_size + j * jcp_.blk_size;

            load_scalar(xmm_tmp, ptr[reg_src + offset * jcp_.data_size], data_type);
            store_scalar(ptr[reg_prc + (i * jcp_.sort_stride + j) * jcp_.data_size], xmm_tmp, data_type);

            uni_vmovdqu(xmm_tmp, table_val(i));
            store_scalar(ptr[reg_prc_idx + (i * jcp_.sort_stride + j) * sizeof(int)],
                            xmm_tmp,
                            memory::data_type::s32);
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::bitonic_BLK_on_channel_store(int elt_num) {
    for (int i = 0; i < jcp_.top_k; i++) {
        for (int j = 0; j < elt_num; j++) {
            int offset = i / jcp_.blk_size * blk_stride + i % jcp_.blk_size + j * jcp_.blk_size;

            load_scalar(xmm_tmp, ptr[reg_prc + (i * jcp_.sort_stride + j) * jcp_.data_size], data_type);
            store_scalar(ptr[reg_dst + offset * jcp_.data_size], xmm_tmp, data_type);

            load_scalar(xmm_tmp,
                        ptr[reg_prc_idx + (i * jcp_.sort_stride + j) * sizeof(int)],
                        memory::data_type::s32);
            store_scalar(ptr[reg_dst_idx + offset * sizeof(int)], xmm_tmp, memory::data_type::s32);
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::bitonic_get_addr(Xbyak::Reg64 reg_base, int data_size, int offset = 0) {
    mov(reg_aux_idx.cvt32(), ptr[reg_aux + offset]);
    mul_by_const(reg_aux_idx, reg_tmp_64, data_size);
    add(reg_aux_idx, reg_base);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::bitonic_swap_vector(int elt_num, bool cmp_val = true) {
    bitonic_get_addr(reg_prc, jcp_.data_size, 0);
    load(reg_aux_idx, vmm_val_l, elt_num);

    bitonic_get_addr(reg_prc, jcp_.data_size, sizeof(int));
    load(reg_aux_idx, vmm_val_r, elt_num);

    bitonic_get_addr(reg_prc_idx, sizeof(int), 0);
    load_i32(reg_aux_idx, vmm_idx_l, elt_num);

    bitonic_get_addr(reg_prc_idx, sizeof(int), sizeof(int));
    load_i32(reg_aux_idx, vmm_idx_r, elt_num);

    swap_vector(vmm_val_l, vmm_idx_l, vmm_val_r, vmm_idx_r, cmp_val);

    bitonic_get_addr(reg_prc, jcp_.data_size, 0);
    store(vmm_val_l, reg_aux_idx, elt_num);

    bitonic_get_addr(reg_prc, jcp_.data_size, sizeof(int));
    store(vmm_val_r, reg_aux_idx, elt_num);

    bitonic_get_addr(reg_prc_idx, sizeof(int), 0);
    store_i32(vmm_idx_l, reg_aux_idx, elt_num);

    bitonic_get_addr(reg_prc_idx, sizeof(int), sizeof(int));
    store_i32(vmm_idx_r, reg_aux_idx, elt_num);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::topk_heap_sorting() {
    mov(reg_heap_seq_idx, ptr[reg_params + GET_OFF(idx_seq_buf)]);
    mov(reg_heap_axis_dim, ptr[reg_params + GET_OFF(axis_dim)]);
    mov(reg_heap_top_k, ptr[reg_params + GET_OFF(top_k)]);

    // init dst
    mov(reg_i, 0);
    sub(reg_heap_top_k, vector_step);
    topk_heap_load(reg_heap_k_sub_step, vector_step);
    add(reg_heap_top_k, vector_step);
    topk_heap_load(reg_heap_top_k, 1);
    mov(reg_zero, 0);

    // Heapify the only node, or start from the last non-leaf node to the root,
    Xbyak::Label topk_heapify_set_label;
    Xbyak::Label topk_heapify_set_end_label;
    cmp(reg_heap_top_k, 1);
    jg(topk_heapify_set_label, T_NEAR);
    mov(reg_end, 0);
    jmp(topk_heapify_set_end_label, T_NEAR);
    L(topk_heapify_set_label);
    reg_sub_shr(reg_end, reg_heap_top_k, 2, 1);
    L(topk_heapify_set_end_label);

    // heapify
    Xbyak::Label topk_heapify_loop_label;
    Xbyak::Label topk_heapify_loop_end_label;
    mov(reg_i, reg_end);
    sub(reg_heap_top_k, 1);
    L(topk_heapify_loop_label);
    {
        heapify_sub_tree(reg_i, reg_heap_k_sub_1);

        cmp(reg_i, 0);
        je(topk_heapify_loop_end_label, T_NEAR);

        sub(reg_i, 1);
        jmp(topk_heapify_loop_label, T_NEAR);
    }
    L(topk_heapify_loop_end_label);
    add(reg_heap_top_k, 1);

    // update
    Xbyak::Label topk_main_loop_label;
    Xbyak::Label topk_main_loop_end_label;
    mov(reg_i, reg_heap_top_k);
    sub(reg_heap_top_k, 1);
    L(topk_main_loop_label);
    {
        cmp(reg_i, reg_heap_axis_dim);
        je(topk_main_loop_end_label, T_NEAR);

        Xbyak::Label topk_update_loop_end_label;
        get_addr_by_reg_idx(reg_aux, reg_src, reg_i, jcp_.data_size);
        load_scalar(xmm_val_p, ptr[reg_aux], data_type);

        get_addr_by_reg_idx(reg_aux, reg_heap_seq_idx, reg_i, sizeof(int));
        load_scalar(xmm_idx_p, ptr[reg_aux], memory::data_type::s32);
        load_scalar(xmm_val_l, ptr[reg_dst], data_type);
        load_scalar(xmm_idx_l, ptr[reg_dst_idx], memory::data_type::s32);

        heap_cmp_node(xmm_val_p, xmm_idx_p, xmm_val_l, xmm_idx_l);
        JMP_TO_LABEL(topk_update_loop_end_label);

        store_scalar(ptr[reg_dst], xmm_val_p, data_type);
        store_scalar(ptr[reg_dst_idx], xmm_idx_p, memory::data_type::s32);
        heapify_sub_tree(reg_zero, reg_heap_k_sub_1);

        L(topk_update_loop_end_label);

        add(reg_i, 1);
        jmp(topk_main_loop_label, T_NEAR);
    }
    L(topk_main_loop_end_label);

    // extract topk
    if (jcp_.sort_index) {
        // reheapify by index
        Xbyak::Label topk_reheapify_loop_label;
        Xbyak::Label topk_reheapify_loop_end_label;
        mov(reg_i, reg_end);
        L(topk_reheapify_loop_label);
        {
            heapify_sub_tree(reg_i, reg_heap_k_sub_1, false);

            cmp(reg_i, 0);
            je(topk_reheapify_loop_end_label, T_NEAR);

            sub(reg_i, 1);
            jmp(topk_reheapify_loop_label, T_NEAR);
        }
        L(topk_reheapify_loop_end_label);

        // extract by index
        topk_heap_extract(false);
    } else {
        // extract by value
        topk_heap_extract();
    }
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::topk_heap_load(Xbyak::Reg64& reg_end, int s) {
    Xbyak::Label topk_init_loop_label;
    Xbyak::Label topk_init_loop_end_label;
    L(topk_init_loop_label);
    {
        if (s == vector_step) {
            cmp(reg_i, reg_end);
            jg(topk_init_loop_end_label, T_NEAR);
        } else {
            cmp(reg_i, reg_end);
            je(topk_init_loop_end_label, T_NEAR);
        }

        get_addr_by_reg_idx(reg_heap_outer_aux, reg_src, reg_i, jcp_.data_size);
        load(reg_heap_outer_aux, vmm_tmp, s);

        get_addr_by_reg_idx(reg_heap_outer_aux, reg_dst, reg_i, jcp_.data_size);
        store(vmm_tmp, reg_heap_outer_aux, s);
        if (s == vector_step) {
            table_to_vmm(vmm_tmp, reg_heap_seq_idx, reg_i, 0, sizeof(int));
        } else {
            get_addr_by_reg_idx(reg_heap_outer_aux, reg_heap_seq_idx, reg_i, sizeof(int));
            load_i32(reg_heap_outer_aux, vmm_tmp, 1);
        }
        get_addr_by_reg_idx(reg_heap_outer_aux, reg_dst_idx, reg_i, sizeof(int));
        store_i32(vmm_tmp, reg_heap_outer_aux, s);

        add(reg_i, s);
        jmp(topk_init_loop_label, T_NEAR);
    }
    L(topk_init_loop_end_label);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::topk_heap_extract(bool cmp_val = true) {
    Xbyak::Label topk_extract_label;
    Xbyak::Label topk_extract_end_label;
    mov(reg_i, reg_heap_k_sub_1);
    L(topk_extract_label);
    {
        cmp(reg_i, 0);
        je(topk_extract_end_label, T_NEAR);

        heap_swap_root(reg_i);
        sub(reg_i, 1);
        heapify_sub_tree(reg_zero, reg_i_sub_1, cmp_val);

        jmp(topk_extract_label, T_NEAR);
    }
    L(topk_extract_end_label);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::heapify_sub_tree(const Xbyak::Reg64& reg_idx, const Xbyak::Reg64& reg_valid, bool cmp_val = true) {
    Xbyak::Label topk_heapify_loop_label;
    Xbyak::Label topk_heapify_loop_end_label;
    Xbyak::Label topk_lchild_loop_label;
    Xbyak::Label topk_rchild_loop_label;

    sub(rsp, sizeof(int));
    mov(ptr[rsp], reg_i.cvt32());
    sub(rsp, sizeof(int));
    mov(ptr[rsp], reg_heap_top_k.cvt32());
    sub(rsp, sizeof(int));
    mov(ptr[rsp], reg_zero.cvt32());
    sub(rsp, sizeof(int));
    mov(ptr[rsp], reg_end.cvt32());
    sub(rsp, sizeof(int));
    mov(ptr[rsp], reg_heap_axis_dim.cvt32());
    sub(rsp, sizeof(int64_t));
    mov(ptr[rsp], reg_src);
    sub(rsp, sizeof(int64_t));
    mov(ptr[rsp], reg_heap_seq_idx);
    sub(rsp, sizeof(int64_t));
    mov(ptr[rsp], reg_params);

    // reg_idx and reg_valid should not be used
    mov(reg_heapify_i, reg_idx);
    mov(reg_heapify_valid, reg_valid);

    cmp(reg_heapify_valid, 0);
    je(topk_heapify_loop_end_label, T_NEAR);

    reg_sub_shr(reg_heapify_end, reg_heapify_valid, 1, 1);
    mov(reg_j, reg_heapify_i);
    mov(reg_aux, reg_dst);
    mov(reg_aux_idx, reg_dst_idx);
    reg_mul_add(reg_aux, reg_heapify_tmp, reg_heapify_i, jcp_.data_size);
    reg_mul_add(reg_aux_idx, reg_heapify_tmp, reg_heapify_i, sizeof(int));
    reg_mul_add(reg_offset, reg_heapify_i, 2, 1);
    mul_by_const(reg_offset, reg_heapify_tmp, jcp_.data_size);
    if (jcp_.data_size != sizeof(int)) {
        reg_mul_add(reg_offset_idx, reg_heapify_i, 2, 1);
        mul_by_const(reg_offset_idx, reg_heapify_tmp, sizeof(int));
    }

    L(topk_heapify_loop_label);
    {
        cmp(reg_j, reg_heapify_end);
        jg(topk_heapify_loop_end_label, T_NEAR);

        load_scalar(xmm_val_p, ptr[reg_aux], data_type);
        load_scalar(xmm_idx_p, ptr[reg_aux_idx], memory::data_type::s32);

        // compare lchild-rchild
        mov(reg_prc, reg_dst);
        add(reg_prc, reg_offset);
        mov(reg_prc_idx, reg_dst_idx);
        if (jcp_.data_size != sizeof(int)) {
            add(reg_prc_idx, reg_offset_idx);
        } else {
            add(reg_prc_idx, reg_offset);
        }
        load_scalar(xmm_val_l, ptr[reg_prc], data_type);
        load_scalar(xmm_idx_l, ptr[reg_prc_idx], memory::data_type::s32);
        add(reg_prc, jcp_.data_size);
        add(reg_prc_idx, sizeof(int));

        // if last valid parent has no rchild
        mov(reg_heapify_tmp, reg_heapify_valid);
        shr(reg_heapify_tmp, 1);
        cmp(reg_j, reg_heapify_tmp);
        je(topk_lchild_loop_label, T_NEAR);

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
            add(reg_offset, jcp_.data_size);
            shl(reg_offset, 1);
            add(reg_offset, jcp_.data_size);
            if (jcp_.data_size != sizeof(int)) {
                add(reg_offset_idx, sizeof(int));
                shl(reg_offset_idx, 1);
                add(reg_offset_idx, sizeof(int));
            }
            shl(reg_j, 1);
            add(reg_j, 2);
            jmp(topk_heapify_loop_label, T_NEAR);
        }

        // compare node-lchild
        L(topk_lchild_loop_label);
        {
            heap_cmp_node(xmm_val_p, xmm_idx_p, xmm_val_l, xmm_idx_l, cmp_val);
            JMP_TO_LABEL(topk_heapify_loop_end_label);

            sub(reg_prc, jcp_.data_size);
            sub(reg_prc_idx, sizeof(int));
            heap_swap_node(xmm_val_p, xmm_idx_p, xmm_val_l, xmm_idx_l);
            mov(reg_aux, reg_prc);
            mov(reg_aux_idx, reg_prc_idx);
            shl(reg_offset, 1);
            add(reg_offset, jcp_.data_size);
            if (jcp_.data_size != sizeof(int)) {
                shl(reg_offset_idx, 1);
                add(reg_offset_idx, sizeof(int));
            }
            shl(reg_j, 1);
            add(reg_j, 1);
            jmp(topk_heapify_loop_label, T_NEAR);
        }
    }
    L(topk_heapify_loop_end_label);

    mov(reg_params, ptr[rsp]);
    add(rsp, sizeof(int64_t));
    mov(reg_heap_seq_idx, ptr[rsp]);
    add(rsp, sizeof(int64_t));
    mov(reg_src, ptr[rsp]);
    add(rsp, sizeof(int64_t));
    mov(reg_heap_axis_dim.cvt32(), ptr[rsp]);
    add(rsp, sizeof(int));
    mov(reg_end.cvt32(), ptr[rsp]);
    add(rsp, sizeof(int));
    mov(reg_zero.cvt32(), ptr[rsp]);
    add(rsp, sizeof(int));
    mov(reg_heap_top_k.cvt32(), ptr[rsp]);
    add(rsp, sizeof(int));
    mov(reg_i.cvt32(), ptr[rsp]);
    add(rsp, sizeof(int));
}

template <cpu_isa_t isa>
inline bool jit_uni_topk_kernel_f32<isa>::is_valid_isa(cpu_isa_t cpu_isa) {
    return mayiuse(cpu_isa);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::uni_vpcmpgtd(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2, const Xbyak::Operand& op) {
    if (is_valid_isa(cpu::x64::avx)) {
        vpcmpgtd(x1, x2, op);
    } else {
        if (x1.getIdx() != x2.getIdx()) {
            uni_vmovups(x1, x2);
        }
        pcmpgtd(x1, op);
    }
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::uni_vpcmpgtd(const Xbyak::Ymm& x1, const Xbyak::Ymm& x2, const Xbyak::Operand& op) {
    vpcmpgtd(x1, x2, op);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::compare_node_xmm(Xmm xmm_val_a,
                                Xmm xmm_idx_a,
                                Xmm xmm_val_b,
                                Xmm xmm_idx_b,
                                Xmm mask,
                                unsigned char val_cmp_flg,
                                unsigned char idx_cmp_flg,
                                bool cmp_val) {
    if (isa == cpu::x64::avx512_core) {
        if (cmp_val) {
            if (isFloatCompatible(data_type)) {
                vcmpps(k_mask, xmm_val_a, xmm_val_b, val_cmp_flg);
            } else {
                vpcmpd(k_mask, xmm_val_a, xmm_val_b, val_cmp_flg);
            }
        } else {
            vpcmpd(k_mask, xmm_idx_a, xmm_idx_b, idx_cmp_flg);
        }
    } else {
        if (cmp_val) {
            if (isFloatCompatible(data_type)) {
                uni_vcmpps(mask, xmm_val_a, xmm_val_b, val_cmp_flg);
            } else {
                if (val_cmp_flg == _cmp_nle_us) {
                    uni_vpcmpgtd(mask, xmm_val_a, xmm_val_b);
                } else {
                    uni_vpcmpgtd(mask, xmm_val_b, xmm_val_a);
                }
            }
        } else {
            if (idx_cmp_flg == _cmp_nle_us) {
                uni_vpcmpgtd(mask, xmm_idx_a, xmm_idx_b);
            } else {
                uni_vpcmpgtd(mask, xmm_idx_b, xmm_idx_a);
            }
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::heap_cmp_node(Xmm xmm_val_a, Xmm xmm_idx_a, Xmm xmm_val_b, Xmm xmm_idx_b, bool cmp_val = true) {
    compare_node_xmm(xmm_val_a, xmm_idx_a, xmm_val_b, xmm_idx_b, xmm_mask, heap_cmp_flg, _cmp_lt_os, cmp_val);
}

// n: node, c: child
template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::heap_swap_node(Xmm xmm_val_n, Xmm xmm_idx_n, Xmm xmm_val_c, Xmm xmm_idx_c) {
    // swap store
    store_scalar(ptr[reg_aux], xmm_val_c, data_type);
    store_scalar(ptr[reg_aux_idx], xmm_idx_c, memory::data_type::s32);
    store_scalar(ptr[reg_prc], xmm_val_n, data_type);
    store_scalar(ptr[reg_prc_idx], xmm_idx_n, memory::data_type::s32);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::heap_swap_root(const Xbyak::Reg64& reg_idx) {
    get_addr_by_reg_idx(reg_aux, reg_dst, reg_idx, jcp_.data_size);
    get_addr_by_reg_idx(reg_aux_idx, reg_dst_idx, reg_idx, sizeof(int));

    load_scalar(xmm_val_p, ptr[reg_dst], data_type);
    load_scalar(xmm_idx_p, ptr[reg_dst_idx], memory::data_type::s32);
    load_scalar(xmm_val_l, ptr[reg_aux], data_type);
    load_scalar(xmm_idx_l, ptr[reg_aux_idx], memory::data_type::s32);

    store_scalar(ptr[reg_aux], xmm_val_p, data_type);
    store_scalar(ptr[reg_aux_idx], xmm_idx_p, memory::data_type::s32);
    store_scalar(ptr[reg_dst], xmm_val_l, data_type);
    store_scalar(ptr[reg_dst_idx], xmm_idx_l, memory::data_type::s32);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::topk_bubble_vector() {
    mov(reg_bubble_block_idx, ptr[reg_params + GET_OFF(idx_block_buf)]);
    if (!jcp_.bubble_inplace) {
        mov(reg_block_sort_stride, ptr[reg_params + GET_OFF(sort_stride)]);
        mov(reg_bubble_axis_dim, ptr[reg_params + GET_OFF(axis_dim)]);
        mov(reg_bubble_block_top_k, ptr[reg_params + GET_OFF(top_k)]);
    }

    Xbyak::Label topk_main_loop_label;
    Xbyak::Label topk_main_loop_end_label;
    L(topk_main_loop_label);
    {
        cmp(reg_work_amount, vector_step);
        jl(topk_main_loop_end_label, T_NEAR);

        if (jcp_.bubble_inplace) {
            topk_bubble_inplace(vector_step);
        } else {
            topk_bubble(vector_step);
        }

        add(reg_src, vector_step * jcp_.data_size);
        add(reg_dst, vector_step * jcp_.data_size);
        add(reg_dst_idx, vector_step * sizeof(int));
        sub(reg_work_amount, vector_step);

        jmp(topk_main_loop_label, T_NEAR);
    }
    L(topk_main_loop_end_label);

    // tail
    if (jcp_.bubble_inplace) {
        if (tail_step) {
            Xbyak::Label topk_tail_loop_end_label;
            cmp(reg_work_amount, tail_step);
            jl(topk_tail_loop_end_label, T_NEAR);

            topk_bubble_inplace(tail_step);

            L(topk_tail_loop_end_label);
        }
    } else {
        Xbyak::Label topk_tail_loop_label;
        Xbyak::Label topk_tail_loop_end_label;
        L(topk_tail_loop_label);
        {
            cmp(reg_work_amount, 0);
            je(topk_tail_loop_end_label, T_NEAR);

            topk_bubble(1);

            add(reg_src, jcp_.data_size);
            add(reg_dst, jcp_.data_size);
            add(reg_dst_idx, sizeof(int));
            sub(reg_work_amount, 1);

            jmp(topk_tail_loop_label, T_NEAR);
        }
        L(topk_tail_loop_end_label);
    }
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::reg_add(const Xbyak::Reg64& reg_sum, const Xbyak::Reg64& reg_a, const Xbyak::Reg64& reg_b) {
    mov(reg_sum, reg_a);
    add(reg_sum, reg_b);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::query_table_by_reg_idx(const Xbyak::Reg64& reg_table,
                                    const Xbyak::Reg64& reg_idx,
                                    int offset,
                                    size_t size) {
    mov(reg_tmp, reg_idx);
    add(reg_tmp, offset);
    mul_by_const(reg_tmp, reg_tmp_64, size);
    add(reg_tmp, reg_table);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::table_to_vmm(Vmm vmm_src,
                            const Xbyak::Reg64& reg_table,
                            const Xbyak::Reg64& reg_idx,
                            int offset,
                            size_t size) {
    query_table_by_reg_idx(reg_table, reg_idx, offset, size);
    uni_vmovdqu(vmm_src, ptr[reg_tmp]);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::table_to_xmm(Xmm xmm_src,
                            const Xbyak::Reg64& reg_table,
                            const Xbyak::Reg64& reg_idx,
                            int offset,
                            size_t size) {
    query_table_by_reg_idx(reg_table, reg_idx, offset, size);
    uni_vmovss(xmm_src, ptr[reg_tmp]);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::get_addr_by_reg_idx(const Xbyak::Reg& reg_out,
                                const Xbyak::Reg& reg_base,
                                const Xbyak::Reg64& reg_in,
                                int value) {
    mov(reg_out, reg_in);
    mul_by_const(reg_out, reg_tmp_64, value);
    add(reg_out, reg_base);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::get_addr_by_reg_idx(const Xbyak::Reg& reg_out,
                                const Xbyak::Reg& reg_base,
                                const Xbyak::Reg64& reg_in,
                                int value,
                                const Xbyak::Reg64& reg_value) {
    mov(reg_out, reg_in);
    imul(reg_out, reg_value);
    mul_by_const(reg_out, reg_tmp_64, value);
    add(reg_out, reg_base);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::get_addr_by_reg_idx(const Xbyak::Reg& reg_out,
                                const Xbyak::Reg& reg_base,
                                const Xbyak::Reg64& reg_in,
                                const Xbyak::Reg64& reg_value) {
    mov(reg_out, reg_in);
    imul(reg_out, reg_value);
    add(reg_out, reg_base);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::reg_mul_add(const Xbyak::Reg& reg_out, const Xbyak::Reg64& reg_in, int mul_val, int add_val) {
    mov(reg_out, reg_in);
    mul_by_const(reg_out, reg_tmp_64, mul_val);
    add(reg_out, add_val);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::reg_mul_add(const Xbyak::Reg& reg_out,
                        const Xbyak::Reg& reg_tmp,
                        const Xbyak::Reg64& reg_in,
                        int mul_val) {
    mov(reg_tmp, reg_in);
    mul_by_const(reg_tmp, reg_tmp_64, mul_val);
    add(reg_out, reg_tmp);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::reg_mul_add(const Xbyak::Reg& reg_out, int mul_val, const Xbyak::Reg64& reg_base) {
    mul_by_const(reg_out, reg_tmp_64, mul_val);
    add(reg_out, reg_base);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::reg_sub_shr(const Xbyak::Reg& reg_out, const Xbyak::Reg64& reg_in, int sub_val, int shr_val) {
    mov(reg_out, reg_in);
    sub(reg_out, sub_val);
    shr(reg_out, shr_val);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::reg_sub_mul(const Xbyak::Reg& reg_out, const Xbyak::Reg64& reg_in, int sub_val, int mul_val) {
    mov(reg_out, reg_in);
    sub(reg_out, sub_val);
    mul_by_const(reg_out, reg_tmp_64, mul_val);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::reg_shl(const Xbyak::Reg& reg_out, int rate) {
    switch (rate) {
    case 1:
        break;
    case 2:
        shl(reg_out, 1);
        break;
    case 4:
        shl(reg_out, 2);
        break;
    default:
        assert(!"incorrect data size rate");
    }
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::reg_shr(const Xbyak::Reg& reg_out, int rate) {
    switch (rate) {
    case 1:
        break;
    case 2:
        shr(reg_out, 1);
        break;
    case 4:
        shr(reg_out, 2);
        break;
    default:
        assert(!"incorrect data size rate");
    }
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::reg_div_blk_size(const Xbyak::Reg& reg_out, const Xbyak::Reg64& reg_in, int blk_size) {
    mov(reg_out, reg_in);
    switch (blk_size) {
    case 8:
        shr(reg_out, 3);
        break;
    case 16:
        shr(reg_out, 4);
        break;
    default:
        assert(!"incorrect blk_size");
    }
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::reg_mod_blk_size(const Xbyak::Reg& reg_out, const Xbyak::Reg64& reg_in, int blk_size) {
    mov(reg_out, reg_in);
    reg_div_blk_size(reg_tmp_64, reg_in, blk_size);
    switch (blk_size) {
    case 8:
        shl(reg_tmp_64, 3);
        break;
    case 16:
        shl(reg_tmp_64, 4);
        break;
    default:
        assert(!"incorrect blk_size");
    }
    sub(reg_out, reg_tmp_64);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::reg_calc_offset_by_channel_idx(const Xbyak::Reg& reg_out,
                                            const Xbyak::Reg64& reg_stride,
                                            const Xbyak::Reg64& reg_channel_idx,
                                            int blk_size) {
    reg_div_blk_size(reg_out, reg_channel_idx, blk_size);
    imul(reg_out, reg_stride);
    reg_mod_blk_size(reg_tmp, reg_channel_idx, blk_size);
    add(reg_out, reg_tmp);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::topk_bubble(int elt_num) {
    reg_shl(reg_block_sort_stride, jcp_.data_size);

    // init dst
    Xbyak::Label topk_init_loop_label;
    Xbyak::Label topk_init_loop_end_label;
    mov(reg_i, 0);
    L(topk_init_loop_label);
    {
        cmp(reg_i, reg_bubble_block_top_k);
        je(topk_init_loop_end_label, T_NEAR);

        get_addr_by_reg_idx(reg_tmp, reg_src, reg_block_sort_stride_byte, reg_i);
        load(reg_tmp, vmm_tmp, elt_num);
        get_addr_by_reg_idx(reg_tmp, reg_dst, reg_block_sort_stride_byte, reg_i);
        store(vmm_tmp, reg_tmp, elt_num);

        table_to_vmm(vmm_tmp, reg_bubble_block_idx, reg_i, 0, vlen);
        get_addr_by_reg_idx(reg_tmp, reg_dst_idx, reg_block_sort_stride_byte, sizeof(int) / jcp_.data_size, reg_i);
        store_i32(vmm_tmp, reg_tmp, elt_num);

        add(reg_i, 1);
        jmp(topk_init_loop_label, T_NEAR);
    }
    L(topk_init_loop_end_label);

    // sort
    topk_bubble_vector_sort(elt_num);

    // update
    Xbyak::Label topk_update_loop_label;
    Xbyak::Label topk_update_loop_end_label;
    mov(reg_i, reg_bubble_block_top_k);
    L(topk_update_loop_label);
    {
        cmp(reg_i, reg_bubble_axis_dim);
        je(topk_update_loop_end_label, T_NEAR);

        get_addr_by_reg_idx(reg_tmp, reg_src, reg_block_sort_stride_byte, reg_i);
        load(reg_tmp, vmm_val_r, elt_num);

        table_to_vmm(vmm_idx_r, reg_bubble_block_idx, reg_i, 0, vlen);

        sub(rsp, sizeof(int64_t));
        mov(ptr[rsp], reg_bubble_block_idx);
        sub(rsp, sizeof(int));
        mov(ptr[rsp], reg_bubble_axis_dim.cvt32());

        Xbyak::Label topk_update_inner_loop_label;
        Xbyak::Label topk_update_inner_loop_end_label;
        mov(reg_block_r, reg_bubble_block_top_k);
        sub(reg_bubble_block_top_k, 1);
        L(topk_update_inner_loop_label);
        {
            cmp(reg_block_r, 0);
            je(topk_update_inner_loop_end_label, T_NEAR);

            mov(reg_block_l, reg_block_r);
            sub(reg_block_l, 1);
            bubble_swap_vector(reg_block_l, reg_block_r, elt_num);

            sub(reg_block_r, 1);
            jmp(topk_update_inner_loop_label, T_NEAR);
        }
        L(topk_update_inner_loop_end_label);
        add(reg_bubble_block_top_k, 1);

        mov(reg_bubble_axis_dim.cvt32(), ptr[rsp]);
        add(rsp, sizeof(int));
        mov(reg_bubble_block_idx, ptr[rsp]);
        add(rsp, sizeof(int64_t));

        add(reg_i, 1);
        jmp(topk_update_loop_label, T_NEAR);
    }
    L(topk_update_loop_end_label);

    if (jcp_.sort_index) {
        topk_bubble_vector_sort(elt_num, false);
    }

    reg_shr(reg_block_sort_stride, jcp_.data_size);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::topk_bubble_vector_sort(int elt_num, bool cmp_val = true) {
    sub(rsp, sizeof(int64_t));
    mov(ptr[rsp], reg_bubble_block_idx);
    sub(rsp, sizeof(int));
    mov(ptr[rsp], reg_bubble_axis_dim.cvt32());

    Xbyak::Label topk_sort_loop_label;
    Xbyak::Label topk_sort_loop_end_label;
    mov(reg_i, 0);
    sub(reg_bubble_block_top_k, 1);
    L(topk_sort_loop_label);
    {
        cmp(reg_i, reg_bubble_block_k_sub_1);
        je(topk_sort_loop_end_label, T_NEAR);

        Xbyak::Label topk_sort_inner_loop_label;
        Xbyak::Label topk_sort_inner_loop_end_label;
        mov(reg_block_r, reg_bubble_block_k_sub_1);
        L(topk_sort_inner_loop_label);
        {
            cmp(reg_block_r, reg_i);
            je(topk_sort_inner_loop_end_label, T_NEAR);

            mov(reg_block_l, reg_block_r);
            sub(reg_block_l, 1);
            bubble_swap_vector(reg_block_l, reg_block_r, elt_num, cmp_val);

            sub(reg_block_r, 1);
            jmp(topk_sort_inner_loop_label, T_NEAR);
        }
        L(topk_sort_inner_loop_end_label);

        add(reg_i, 1);
        jmp(topk_sort_loop_label, T_NEAR);
    }
    L(topk_sort_loop_end_label);
    add(reg_bubble_block_top_k, 1);

    mov(reg_bubble_axis_dim.cvt32(), ptr[rsp]);
    add(rsp, sizeof(int));
    mov(reg_bubble_block_idx, ptr[rsp]);
    add(rsp, sizeof(int64_t));
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::topk_bubble_inplace(int elt_num) {
    // load
    for (int i = 0; i < jcp_.top_k; i++) {
        load(reg_src, vmm_val(i), elt_num, i * jcp_.sort_stride * jcp_.data_size);
        uni_vmovdqu(vmm_idx(i), table_val(i));
    }
    // sort
    for (int i = 0; i < jcp_.top_k - 1; i++) {
        for (int j = jcp_.top_k - 1; j > i; j--) {
            swap_vector(vmm_val(j - 1), vmm_idx(j - 1), vmm_val(j), vmm_idx(j));
        }
    }
    for (int i = jcp_.top_k; i < jcp_.axis_dim; i++) {
        load(reg_src, vmm_val(jcp_.top_k), elt_num, i * jcp_.sort_stride * jcp_.data_size);
        uni_vmovdqu(vmm_idx(jcp_.top_k), table_val(i));
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
        store(vmm_val(i), reg_dst, elt_num, i * jcp_.sort_stride * jcp_.data_size);
        store_i32(vmm_idx(i), reg_dst_idx, elt_num, i * jcp_.sort_stride * sizeof(int));
    }
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::topk_bubble_horiz() {
    mov(reg_bubble_axis_dim, ptr[reg_params + GET_OFF(axis_dim)]);
    mov(reg_seq_sort_stride, ptr[reg_params + GET_OFF(sort_stride)]);
    mov(reg_bubble_seq_idx, ptr[reg_params + GET_OFF(idx_seq_buf)]);

    // load and sort
    mov(reg_i, 0);
    Xbyak::Label topk_load_sort_label;
    Xbyak::Label topk_load_sort_end_label;
    cmp(reg_bubble_axis_dim, jcp_.blk_size);
    jge(topk_load_sort_label, T_NEAR);

    load_scalar(xmm_val(0), ptr[reg_src], data_type);
    uni_vmovss(xmm_idx(0), table_bubble_seq_idx(0));
    jmp(topk_load_sort_end_label, T_NEAR);

    L(topk_load_sort_label);
    {
        load(reg_src, vmm_val(0), vector_step, 0);
        uni_vmovdqu(vmm_idx(0), table_bubble_seq_idx(0));
        if (isa == cpu::x64::sse41) {
            load(reg_src, vmm_val(1), vector_step, 4 * jcp_.data_size);
            uni_vmovdqu(vmm_idx(1), table_bubble_seq_idx(4));
            swap_vector(vmm_val(0), vmm_idx(0), vmm_val(1), vmm_idx(1));
        }

        Xbyak::Label topk_iter_label;
        Xbyak::Label topk_iter_end_label;
        mov(reg_i, jcp_.blk_size);
        sub(reg_bubble_axis_dim, jcp_.blk_size);
        L(topk_iter_label);
        {
            cmp(reg_i, reg_bubble_axis_dim);
            jg(topk_iter_end_label, T_NEAR);

            get_addr_by_reg_idx(reg_aux, reg_src, reg_i, jcp_.data_size, reg_seq_sort_stride);
            load(reg_aux, vmm_val(1), vector_step);
            table_to_vmm(vmm_idx(1), reg_bubble_seq_idx, reg_i, 0, sizeof(int));
            swap_vector(vmm_val(0), vmm_idx(0), vmm_val(1), vmm_idx(1));
            if (isa == cpu::x64::sse41) {
                add(reg_aux, 4 * jcp_.data_size);
                load(reg_aux, vmm_val(1), vector_step);
                table_to_vmm(vmm_idx(1), reg_bubble_seq_idx, reg_i, 4, sizeof(int));
                swap_vector(vmm_val(0), vmm_idx(0), vmm_val(1), vmm_idx(1));
            }

            add(reg_i, jcp_.blk_size);
            jmp(topk_iter_label, T_NEAR);
        }
        L(topk_iter_end_label);
        add(reg_bubble_axis_dim, jcp_.blk_size);

        horiz_process();
    }
    L(topk_load_sort_end_label);

    Xbyak::Label topk_tail_label;
    Xbyak::Label topk_tail_end_label;
    get_addr_by_reg_idx(reg_aux, reg_src, reg_i, jcp_.data_size, reg_seq_sort_stride);
    L(topk_tail_label);
    {
        cmp(reg_i, reg_bubble_axis_dim);
        je(topk_tail_end_label, T_NEAR);

        load_scalar(xmm_val(1), ptr[reg_aux], data_type);
        table_to_xmm(xmm_idx(1), reg_bubble_seq_idx, reg_i, 0, sizeof(int));
        bubble_swap_xmm(xmm_val(0), xmm_idx(0), xmm_val(1), xmm_idx(1));

        add(reg_i, 1);
        add(reg_aux, jcp_.data_size);
        jmp(topk_tail_label, T_NEAR);
    }
    L(topk_tail_end_label);

    // store
    store_scalar(ptr[reg_dst], xmm_val(0), data_type);
    store_scalar(ptr[reg_dst_idx], xmm_idx(0), memory::data_type::s32);
}

// dst: xmm_val(0) and xmm_idx(0)
// aux: xmm_val(2/3/4) and xmm_idx(2/3/4)
template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::horiz_process() {
    if (isa == cpu::x64::sse41) {
        horize_top1();
    } else if (isa == cpu::x64::avx2) {
        Xbyak::Ymm ymm_val_dst = Xbyak::Ymm(vmm_val(0).getIdx());
        vextractf128(xmm_val(2), ymm_val_dst, 0);
        vextractf128(xmm_val(3), ymm_val_dst, 1);
        Xbyak::Ymm ymm_idx_dst = Xbyak::Ymm(vmm_idx(0).getIdx());
        vextractf128(xmm_idx(2), ymm_idx_dst, 0);
        vextractf128(xmm_idx(3), ymm_idx_dst, 1);
        bubble_swap_xmm(xmm_val(2), xmm_idx(2), xmm_val(3), xmm_idx(3));
        uni_vmovups(xmm_val(0), xmm_val(2));
        uni_vmovups(xmm_idx(0), xmm_idx(2));
        horize_top1();
    } else {
        Xbyak::Zmm zmm_val_dst = Xbyak::Zmm(vmm_val(0).getIdx());
        vextractf32x4(xmm_val(2), zmm_val_dst, 0);
        vextractf32x4(xmm_val(3), zmm_val_dst, 1);
        Xbyak::Zmm zmm_idx_dst = Xbyak::Zmm(vmm_idx(0).getIdx());
        vextractf32x4(xmm_idx(2), zmm_idx_dst, 0);
        vextractf32x4(xmm_idx(3), zmm_idx_dst, 1);
        bubble_swap_xmm(xmm_val(2), xmm_idx(2), xmm_val(3), xmm_idx(3));
        vextractf32x4(xmm_val(3), zmm_val_dst, 2);
        vextractf32x4(xmm_val(4), zmm_val_dst, 3);
        vextractf32x4(xmm_idx(3), zmm_idx_dst, 2);
        vextractf32x4(xmm_idx(4), zmm_idx_dst, 3);
        bubble_swap_xmm(xmm_val(3), xmm_idx(3), xmm_val(4), xmm_idx(4));
        bubble_swap_xmm(xmm_val(2), xmm_idx(2), xmm_val(3), xmm_idx(3));
        uni_vmovups(xmm_val(0), xmm_val(2));
        uni_vmovups(xmm_idx(0), xmm_idx(2));
        horize_top1();
    }
}

// dst: xmm_val(0) and xmm_idx(0)
// aux: xmm_val(3) and xmm_idx(3)
template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::horize_top1() {
    uni_vmovshdup(xmm_val(3), xmm_val(0));  // dst:1,2,3,4; aux:2,2,4,4
    uni_vmovshdup(xmm_idx(3), xmm_idx(0));
    bubble_swap_xmm(xmm_val(0), xmm_idx(0), xmm_val(3), xmm_idx(3));  // dst:f(1,2),f(2,2),f(3,4),f(4,4)
    uni_vmovhlps(xmm_val(3), xmm_val(3), xmm_val(0));                 // aux:f(3,4),f(4,4),4,4
    uni_vmovhlps(xmm_idx(3), xmm_idx(3), xmm_idx(0));
    bubble_swap_xmm(xmm_val(0), xmm_idx(0), xmm_val(3), xmm_idx(3));  // dst:f(1,2,3,4),...
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::topk_bubble_BLK_on_channel_verti() {
    if (jcp_.bubble_inplace) {
        topk_bubble_BLK_on_channel_inplace();
    } else {
        topk_bubble_BLK_on_channel();
    }
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::topk_bubble_BLK_on_channel() {
    mov(reg_bubble_seq_idx, ptr[reg_params + GET_OFF(idx_seq_buf)]);
    mov(reg_bubble_axis_dim, ptr[reg_params + GET_OFF(axis_dim)]);
    mov(reg_seq_sort_stride, ptr[reg_params + GET_OFF(sort_stride)]);
    mov(reg_bubble_seq_top_k, ptr[reg_params + GET_OFF(top_k)]);

    mul_by_const(reg_seq_sort_stride, reg_tmp_64, jcp_.blk_size);

    // init dst
    Xbyak::Label topk_init_next_label;
    Xbyak::Label topk_init_iter_label;
    Xbyak::Label topk_init_iter_end_label;
    mov(reg_i, 0);
    mov(reg_sub_idx, 0);
    mov(reg_aux, 0);
    L(topk_init_iter_label);
    {
        cmp(reg_i, reg_bubble_seq_top_k);
        je(topk_init_iter_end_label, T_NEAR);

        reg_add(reg_tmp, reg_src, reg_aux);
        load_scalar(xmm_tmp, ptr[reg_tmp], data_type);
        reg_add(reg_tmp, reg_dst, reg_aux);
        store_scalar(ptr[reg_tmp], xmm_tmp, data_type);

        table_to_xmm(xmm_tmp, reg_bubble_seq_idx, reg_i, 0, sizeof(int));
        get_addr_by_reg_idx(reg_tmp, reg_dst_idx, reg_aux, sizeof(int) / jcp_.data_size);
        store_scalar(ptr[reg_tmp], xmm_tmp, memory::data_type::s32);

        add(reg_sub_idx, 1);
        cmp(reg_sub_idx, jcp_.blk_size);
        jl(topk_init_next_label, T_NEAR);

        mov(reg_sub_idx, 0);
        reg_sub_mul(reg_tmp, reg_blk_stride, jcp_.blk_size, jcp_.data_size);
        add(reg_aux, reg_tmp);

        L(topk_init_next_label);
        {
            add(reg_i, 1);
            add(reg_aux, jcp_.data_size);
            jmp(topk_init_iter_label, T_NEAR);
        }
    }
    L(topk_init_iter_end_label);

    // sort
    topk_bubble_BLK_on_channel_sort();

    // update
    Xbyak::Label topk_next_label;
    Xbyak::Label topk_iter_label;
    Xbyak::Label topk_iter_end_label;
    mov(reg_i, reg_bubble_seq_top_k);
    reg_calc_offset_by_channel_idx(reg_seq_tmp, reg_blk_stride, reg_bubble_seq_top_k, jcp_.blk_size);
    get_addr_by_reg_idx(reg_aux, reg_src, reg_seq_tmp, jcp_.data_size);
    reg_mod_blk_size(reg_sub_idx, reg_bubble_seq_top_k, jcp_.blk_size);
    L(topk_iter_label);
    {
        cmp(reg_i, reg_bubble_axis_dim);
        je(topk_iter_end_label, T_NEAR);

        load_scalar(xmm_val_r, ptr[reg_aux], data_type);
        table_to_xmm(xmm_idx_r, reg_bubble_seq_idx, reg_i, 0, sizeof(int));

        sub(rsp, sizeof(int));
        mov(ptr[rsp], reg_prc.cvt32());

        Xbyak::Label topk_update_inner_loop_label;
        Xbyak::Label topk_update_inner_loop_end_label;
        mov(reg_seq_r, reg_bubble_seq_top_k);
        sub(reg_bubble_seq_top_k, 1);
        L(topk_update_inner_loop_label);
        {
            cmp(reg_seq_r, 0);
            je(topk_update_inner_loop_end_label, T_NEAR);

            mov(reg_seq_l, reg_seq_r);
            sub(reg_seq_l, 1);
            bubble_swap_by_index(reg_seq_l, reg_seq_r);

            sub(reg_seq_r, 1);
            jmp(topk_update_inner_loop_label, T_NEAR);
        }
        L(topk_update_inner_loop_end_label);
        add(reg_bubble_seq_top_k, 1);

        mov(reg_prc.cvt32(), ptr[rsp]);
        add(rsp, sizeof(int));

        add(reg_sub_idx, 1);
        cmp(reg_sub_idx, jcp_.blk_size);
        jl(topk_next_label, T_NEAR);

        mov(reg_sub_idx, 0);
        reg_sub_mul(reg_tmp, reg_blk_stride, jcp_.blk_size, jcp_.data_size);
        add(reg_aux, reg_tmp);

        L(topk_next_label);
        {
            add(reg_i, 1);
            add(reg_aux, jcp_.data_size);
            jmp(topk_iter_label, T_NEAR);
        }
    }
    L(topk_iter_end_label);

    if (jcp_.sort_index) {
        topk_bubble_BLK_on_channel_sort(false);
    }
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::topk_bubble_BLK_on_channel_sort(bool cmp_val = true) {
    sub(rsp, sizeof(int));
    mov(ptr[rsp], reg_prc.cvt32());

    Xbyak::Label topk_sort_loop_label;
    Xbyak::Label topk_sort_loop_end_label;
    mov(reg_i, 0);
    sub(reg_bubble_seq_top_k, 1);
    L(topk_sort_loop_label);
    {
        cmp(reg_i, reg_bubble_seq_k_sub_1);
        je(topk_sort_loop_end_label, T_NEAR);

        Xbyak::Label topk_sort_inner_loop_label;
        Xbyak::Label topk_sort_inner_loop_end_label;
        mov(reg_seq_r, reg_bubble_seq_k_sub_1);
        L(topk_sort_inner_loop_label);
        {
            cmp(reg_seq_r, reg_i);
            je(topk_sort_inner_loop_end_label, T_NEAR);

            mov(reg_seq_l, reg_seq_r);
            sub(reg_seq_l, 1);
            bubble_swap_by_index(reg_seq_l, reg_seq_r, cmp_val);

            sub(reg_seq_r, 1);
            jmp(topk_sort_inner_loop_label, T_NEAR);
        }
        L(topk_sort_inner_loop_end_label);

        add(reg_i, 1);
        jmp(topk_sort_loop_label, T_NEAR);
    }
    L(topk_sort_loop_end_label);
    add(reg_bubble_seq_top_k, 1);

    mov(reg_prc.cvt32(), ptr[rsp]);
    add(rsp, sizeof(int));
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::topk_bubble_BLK_on_channel_inplace() {
    // load
    for (int i = 0; i < jcp_.top_k; i++) {
        int offset = i / jcp_.blk_size * blk_stride + i % jcp_.blk_size;
        load_scalar(xmm_val(i), ptr[reg_src + offset * jcp_.data_size], data_type);
        uni_vmovdqu(xmm_idx(i), table_val(i));
    }
    // sort
    for (int i = 0; i < jcp_.top_k - 1; i++) {
        for (int j = jcp_.top_k - 1; j > i; j--) {
            bubble_swap_xmm(xmm_val(j - 1), xmm_idx(j - 1), xmm_val(j), xmm_idx(j));
        }
    }
    for (int i = jcp_.top_k; i < jcp_.axis_dim; i++) {
        int offset = i / jcp_.blk_size * blk_stride + i % jcp_.blk_size;
        load_scalar(xmm_val(jcp_.top_k), ptr[reg_src + offset * jcp_.data_size], data_type);
        uni_vmovdqu(xmm_idx(jcp_.top_k), table_val(i));
        for (int j = jcp_.top_k; j > 0; j--) {
            bubble_swap_xmm(xmm_val(j - 1), xmm_idx(j - 1), xmm_val(j), xmm_idx(j));
        }
    }
    if (jcp_.sort_index) {
        for (int i = 0; i < jcp_.top_k - 1; i++) {
            for (int j = jcp_.top_k - 1; j > i; j--) {
                bubble_swap_xmm(xmm_val(j - 1), xmm_idx(j - 1), xmm_val(j), xmm_idx(j), false);
            }
        }
    }
    // store
    for (int i = 0; i < jcp_.top_k; i++) {
        int offset = i / jcp_.blk_size * blk_stride + i % jcp_.blk_size;
        store_scalar(ptr[reg_dst + offset * jcp_.data_size], xmm_val(i), data_type);
        store_scalar(ptr[reg_dst_idx + offset * sizeof(int)], xmm_idx(i), memory::data_type::s32);
    }
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::bubble_swap_vector(const Xbyak::Reg64& reg_l,
                                const Xbyak::Reg64& reg_r,
                                int elt_num,
                                bool cmp_val = true) {
    mov(reg_tmp_64, reg_block_sort_stride_byte);
    imul(reg_tmp_64, reg_l);

    // load l
    mov(reg_tmp, reg_tmp_64);
    add(reg_tmp, reg_dst);
    load(reg_tmp, vmm_val_l, elt_num);

    reg_shl(reg_tmp_64, sizeof(int) / jcp_.data_size);
    mov(reg_tmp, reg_tmp_64);
    add(reg_tmp, reg_dst_idx);
    reg_shr(reg_tmp_64, sizeof(int) / jcp_.data_size);
    load_i32(reg_tmp, vmm_idx_l, elt_num);

    // load r
    Xbyak::Label topk_load_jmp_label;
    cmp(reg_r, reg_bubble_block_k_sub_1);
    jg(topk_load_jmp_label, T_NEAR);
    {
        add(reg_tmp_64, reg_block_sort_stride_byte);
        mov(reg_tmp, reg_tmp_64);
        add(reg_tmp, reg_dst);
        load(reg_tmp, vmm_val_r, elt_num);

        reg_shl(reg_tmp_64, sizeof(int) / jcp_.data_size);
        mov(reg_tmp, reg_tmp_64);
        add(reg_tmp, reg_dst_idx);
        reg_shr(reg_tmp_64, sizeof(int) / jcp_.data_size);
        load_i32(reg_tmp, vmm_idx_r, elt_num);

        sub(reg_tmp_64, reg_block_sort_stride_byte);
    }
    L(topk_load_jmp_label);

    swap_vector(vmm_val_l, vmm_idx_l, vmm_val_r, vmm_idx_r, cmp_val);

    // store l
    mov(reg_tmp, reg_tmp_64);
    add(reg_tmp, reg_dst);
    store(vmm_val_l, reg_tmp, elt_num);

    reg_shl(reg_tmp_64, sizeof(int) / jcp_.data_size);
    mov(reg_tmp, reg_tmp_64);
    add(reg_tmp, reg_dst_idx);
    reg_shr(reg_tmp_64, sizeof(int) / jcp_.data_size);
    store_i32(vmm_idx_l, reg_tmp, elt_num);

    // store r
    Xbyak::Label topk_store_jmp_label;
    cmp(reg_r, reg_bubble_block_k_sub_1);
    jg(topk_store_jmp_label, T_NEAR);
    {
        add(reg_tmp_64, reg_block_sort_stride_byte);
        mov(reg_tmp, reg_tmp_64);
        add(reg_tmp, reg_dst);
        store(vmm_val_r, reg_tmp, elt_num);

        reg_shl(reg_tmp_64, sizeof(int) / jcp_.data_size);
        mov(reg_tmp, reg_tmp_64);
        add(reg_tmp, reg_dst_idx);
        reg_shr(reg_tmp_64, sizeof(int) / jcp_.data_size);
        store_i32(vmm_idx_r, reg_tmp, elt_num);
    }
    L(topk_store_jmp_label);
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::swap_vector(Vmm vmm_val_a, Vmm vmm_idx_a, Vmm vmm_val_b, Vmm vmm_idx_b, bool cmp_val = true) {
    compare_node_xmm(vmm_val_a, vmm_idx_a, vmm_val_b, vmm_idx_b, vmm_mask, cmp_flg, _cmp_nle_us, cmp_val);

    if (isa == cpu::x64::avx512_core) {
        uni_vmovups(vmm_tmp, vmm_val_a);
        vblendmps(vmm_val_a | k_mask, vmm_val_a, vmm_val_b);
        vblendmps(vmm_val_b | k_mask, vmm_val_b, vmm_tmp);

        uni_vmovups(vmm_tmp, vmm_idx_a);
        vblendmps(vmm_idx_a | k_mask, vmm_idx_a, vmm_idx_b);
        vblendmps(vmm_idx_b | k_mask, vmm_idx_b, vmm_tmp);
    } else {
        uni_vmovups(vmm_tmp, vmm_val_a);
        uni_vblendvps(vmm_val_a, vmm_val_a, vmm_val_b, vmm_mask);
        uni_vblendvps(vmm_val_b, vmm_val_b, vmm_tmp, vmm_mask);

        uni_vmovups(vmm_tmp, vmm_idx_a);
        uni_vblendvps(vmm_idx_a, vmm_idx_a, vmm_idx_b, vmm_mask);
        uni_vblendvps(vmm_idx_b, vmm_idx_b, vmm_tmp, vmm_mask);
    }
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::bubble_swap_by_index(const Xbyak::Reg64& reg_l, const Xbyak::Reg64& reg_r, bool cmp_val = true) {
    sub(rsp, sizeof(int));
    mov(ptr[rsp], reg_i.cvt32());
    sub(rsp, sizeof(int));
    mov(ptr[rsp], reg_prc_idx.cvt32());
    sub(rsp, sizeof(int64_t));
    mov(ptr[rsp], reg_aux);

    mov(reg_aux, reg_blk_stride);
    reg_calc_offset_by_channel_idx(reg_offset_l, reg_aux, reg_l, jcp_.blk_size);
    reg_calc_offset_by_channel_idx(reg_offset_r, reg_aux, reg_r, jcp_.blk_size);

    get_addr_by_reg_idx(reg_aux, reg_dst, reg_offset_l, jcp_.data_size);
    load_scalar(xmm_val_l, ptr[reg_aux], data_type);
    get_addr_by_reg_idx(reg_aux, reg_dst_idx, reg_offset_l, sizeof(int));
    load_scalar(xmm_idx_l, ptr[reg_aux], memory::data_type::s32);

    Xbyak::Label topk_load_jmp_label;
    cmp(reg_r, reg_bubble_seq_k_sub_1);
    jg(topk_load_jmp_label, T_NEAR);
    {
        get_addr_by_reg_idx(reg_aux, reg_dst, reg_offset_r, jcp_.data_size);
        load_scalar(xmm_val_r, ptr[reg_aux], data_type);
        get_addr_by_reg_idx(reg_aux, reg_dst_idx, reg_offset_r, sizeof(int));
        load_scalar(xmm_idx_r, ptr[reg_aux], memory::data_type::s32);
    }
    L(topk_load_jmp_label);

    bubble_swap_xmm(xmm_val_l, xmm_idx_l, xmm_val_r, xmm_idx_r, cmp_val);

    get_addr_by_reg_idx(reg_aux, reg_dst, reg_offset_l, jcp_.data_size);
    store_scalar(ptr[reg_aux], xmm_val_l, data_type);
    get_addr_by_reg_idx(reg_aux, reg_dst_idx, reg_offset_l, sizeof(int));
    store_scalar(ptr[reg_aux], xmm_idx_l, memory::data_type::s32);

    Xbyak::Label topk_store_jmp_label;
    cmp(reg_r, reg_bubble_seq_k_sub_1);
    jg(topk_store_jmp_label, T_NEAR);
    {
        get_addr_by_reg_idx(reg_aux, reg_dst, reg_offset_r, jcp_.data_size);
        store_scalar(ptr[reg_aux], xmm_val_r, data_type);
        get_addr_by_reg_idx(reg_aux, reg_dst_idx, reg_offset_r, sizeof(int));
        store_scalar(ptr[reg_aux], xmm_idx_r, memory::data_type::s32);
    }
    L(topk_store_jmp_label);

    mov(reg_aux, ptr[rsp]);
    add(rsp, sizeof(int64_t));
    mov(reg_prc_idx.cvt32(), ptr[rsp]);
    add(rsp, sizeof(int));
    mov(reg_i.cvt32(), ptr[rsp]);
    add(rsp, sizeof(int));
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::bubble_swap_xmm(Xmm xmm_val_a, Xmm xmm_idx_a, Xmm xmm_val_b, Xmm xmm_idx_b, bool cmp_val = true) {
    compare_node_xmm(xmm_val_a, xmm_idx_a, xmm_val_b, xmm_idx_b, xmm_mask, cmp_flg, _cmp_nle_us, cmp_val);

    if (isa == cpu::x64::avx512_core) {
        uni_vmovups(xmm_tmp, xmm_val_a);
        vblendmps(xmm_val_a | k_mask, xmm_val_a, xmm_val_b);
        vblendmps(xmm_val_b | k_mask, xmm_val_b, xmm_tmp);

        uni_vmovups(xmm_tmp, xmm_idx_a);
        vblendmps(xmm_idx_a | k_mask, xmm_idx_a, xmm_idx_b);
        vblendmps(xmm_idx_b | k_mask, xmm_idx_b, xmm_tmp);
    } else {
        uni_vmovups(xmm_tmp, xmm_val_a);
        uni_vblendvps(xmm_val_a, xmm_val_a, xmm_val_b, xmm_mask);
        uni_vblendvps(xmm_val_b, xmm_val_b, xmm_tmp, xmm_mask);

        uni_vmovups(xmm_tmp, xmm_idx_a);
        uni_vblendvps(xmm_idx_a, xmm_idx_a, xmm_idx_b, xmm_mask);
        uni_vblendvps(xmm_idx_b, xmm_idx_b, xmm_tmp, xmm_mask);
    }
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::load_scalar(Xmm xmm_src, const Xbyak::Address& op, memory::data_type src_dt, bool cvt_dt = false) {
    switch (src_dt) {
    case memory::data_type::f32:
    case memory::data_type::s32:
        uni_vmovss(xmm_src, op);
        break;
    case memory::data_type::bf16:
        uni_vpinsrw(xmm_src, xmm_src, op, 0x0);
        uni_vpslld(xmm_src, xmm_src, 16);
        break;
    case memory::data_type::s8:
        movsx(reg_tmp_32, op);
        uni_vmovq(xmm_src, reg_tmp_64);
        break;
    case memory::data_type::u8:
        movzx(reg_tmp_32, op);
        uni_vmovq(xmm_src, reg_tmp_64);
        break;
    default:
        assert(!"unknown src_dt");
    }

    if (cvt_dt && !isFloatCompatible(src_dt)) {
        uni_vcvtdq2ps(xmm_src, xmm_src);
    }
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::store_scalar(const Xbyak::Address& op, Xmm xmm_dst, memory::data_type dst_dt, bool cvt_dt = false) {
    if (cvt_dt && !isFloatCompatible(dst_dt)) {
        uni_vcvtps2dq(xmm_dst, xmm_dst);
    }

    switch (dst_dt) {
    case memory::data_type::f32:
    case memory::data_type::s32:
        uni_vmovss(op, xmm_dst);
        break;
    case memory::data_type::bf16:
        uni_vpsrld(xmm_dst, xmm_dst, 16);
        uni_vpextrw(op, xmm_dst, 0x0);
        break;
    case memory::data_type::s8:
        uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
        uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
        uni_vmovq(reg_tmp_64, xmm_dst);
        mov(op, reg_tmp_8);
        break;
    case memory::data_type::u8:
        uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
        uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
        uni_vmovq(reg_tmp_64, xmm_dst);
        mov(op, reg_tmp_8);
        break;
    default:
        assert(!"unknown dst_dt");
    }
}

template <cpu_isa_t isa>
inline void jit_uni_topk_kernel_f32<isa>::prepare_idx_table() {
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
}

template struct jit_uni_topk_kernel_f32<cpu_isa_t::avx512_core>;
template struct jit_uni_topk_kernel_f32<cpu_isa_t::avx2>;
template struct jit_uni_topk_kernel_f32<cpu_isa_t::sse41>;

}  // namespace kernel
}  // namespace intel_cpu
}  // namespace ov
