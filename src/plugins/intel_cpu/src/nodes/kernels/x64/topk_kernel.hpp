// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"
#include "cpu/x64/jit_generator.hpp"
#include "/home/chen/git/parallel/topk_kernel/openvino/src/plugins/intel_cpu/thirdparty/onednn/src/cpu/x64/jit_generator.hpp"
#include "emitters/plugin/x64/jit_emitter.hpp"

namespace ov {
namespace intel_cpu {
namespace kernel {

enum TopKLayoutType { topk_ncsp, topk_nspc, topk_blocked };

enum TopKAlgorithm { topk_bubble_sort, topk_bitonic_sort, topk_heap_sort };

struct jit_topk_config_params {
    bool mode_max;          // which of the two elements to select. ture: max; false: min
    bool sort_index;        // sort by value or index. true: index; false: value
    bool topk_innermost;    // if topk sorting is applied on innermost dimension or other dimension
    bool bubble_inplace;    // all the elements in sorting is right in the register, no need to load and store for each
                            // comparison
    bool stable;            // if require stable sorting
    TopKLayoutType layout;  // memory layout
    TopKAlgorithm algorithm;      // topk sorting algorithm
    ov::element::Type precision;  // precision
    int data_size;                // data size
    int blk_size;                 // block size
    int top_k;                    // number of the output elements in the sorting dimension
    int work_amount;              // how many elements are processed when call jit kernel once
    int axis_dim;                 // size of topk axis
    int sort_stride;              // memory stride of adjacent elements in sorting
    int bitonic_idx_cnt;  // the repeatedly counted total number of elements in sorting, which equal the total number of
                          // comparison x 2
    int bitonic_k_idx_cnt;  // the counterpart of bitonic_idx_cnt, when sort_index == true
};

struct jit_topk_call_args {
    const void* src;
    void* process;
    void* process_index;
    void* dst;
    void* index;
    const int* bitonic_idx_buf;
    const int* bitonic_k_idx_buf;
    const int* idx_block_buf;  // original idx sequence, repeated by block (eg. 00000000,11111111,...,77777777), only
                               // used in bubble sort
    const int* idx_seq_buf;    // original idx sequence (eg. 01234567), only used in bubble sort and heap sort
    size_t axis_dim;  // point to axis_dim, only used in heap sort with dynamic shapes to achieve axis_dim agnosic
    size_t top_k;
    size_t work_amount;
    size_t sort_stride;
};

struct jit_uni_topk_kernel {
    void (*ker_)(const jit_topk_call_args*);

    void operator()(const jit_topk_call_args* args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_topk_kernel(jit_topk_config_params jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_topk_kernel() {}

    virtual void create_ker() = 0;

    jit_topk_config_params jcp_;
};


template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct jit_uni_topk_kernel_f32 : public jit_uni_topk_kernel, public dnnl::impl::cpu::x64::jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_topk_kernel_f32)

    explicit jit_uni_topk_kernel_f32(jit_topk_config_params jcp)
        : jit_uni_topk_kernel(jcp),
          jit_generator(jit_name()) {}

    void create_ker() override;

    void generate() override;

private:
    using Vmm =
        typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41, Xbyak::Xmm, isa == dnnl::impl::cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    size_t vlen = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen;
    dnnl::memory::data_type data_type;
    ov::element::Type precision_in_reg;

    Xbyak::Address table_val(int index) {
        return ptr[reg_table + index * vlen];
    }
    Xbyak::Address table_bubble_block_idx(int index) {
        return ptr[reg_bubble_block_idx + index * vlen];
    }
    Xbyak::Address table_bubble_seq_idx(int index) {
        return ptr[reg_bubble_seq_idx + index * sizeof(int)];
    }
    Xbyak::Address table_heap_seq_idx(int index) {
        return ptr[reg_heap_seq_idx + index * sizeof(int)];
    }

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 reg_dst_idx = r10;
    Xbyak::Reg64 reg_prc = r11;
    Xbyak::Reg64 reg_prc_idx = r12;
    Xbyak::Reg64 reg_work_amount = r13;
    Xbyak::Reg64 reg_table = r14;
    Xbyak::Reg64 reg_params = Xbyak::Reg64(dnnl::impl::cpu::x64::abi_param_regs[0]);
    Xbyak::Reg64 reg_i = rax;
    Xbyak::Reg64 reg_aux = rdx;
    Xbyak::Reg64 reg_aux_idx = rbx;

    Xbyak::Reg8 reg_tmp_8 = r15b;
    Xbyak::Reg32 reg_tmp_32 = r15d;
    Xbyak::Reg64 reg_tmp_64 = r15;

    Xbyak::Reg64 reg_load_table = rbp;
    Xbyak::Reg64 reg_load_store_mask = rsi;

    // ================================================ for shape_agnostic_alg
    // ================================================
    // *** for both heap sort and bubble sort ***
    Xbyak::Reg64 reg_tmp = reg_aux_idx;

    // *** for heap sort only ***
    Xbyak::Reg64 reg_j = reg_i;                // save reg_i by rsp before using reg_j
    Xbyak::Reg64 reg_offset = reg_load_table;  // reuse reg_load_table after finish using load/store_emiter
    Xbyak::Reg64 reg_offset_idx =
        reg_load_store_mask;  // reuse reg_load_store_mask after finish using load/store_emiter
    Xbyak::Reg64 reg_heap_seq_idx = reg_table;
    Xbyak::Reg64 reg_heap_axis_dim = reg_work_amount;
    Xbyak::Reg64 reg_heap_top_k = reg_prc;  // save reg_top_k by rsp before using reg_prc
    Xbyak::Reg64 reg_heap_k_sub_step = reg_heap_top_k;
    Xbyak::Reg64 reg_zero = reg_offset;  // save reg_zero by rsp before using reg_offset, also refer to reg_offset
    Xbyak::Reg64 reg_end = reg_prc_idx;  // save reg_heap_outer_aux by rsp before using reg_prc_idx
    Xbyak::Reg64 reg_heap_outer_aux = reg_prc_idx;
    Xbyak::Reg64 reg_i_sub_1 = reg_i;                   // denotes i-1
    Xbyak::Reg64 reg_heap_k_sub_1 = reg_heap_top_k;     // denotes k-1
    Xbyak::Reg64 reg_heapify_end = reg_heap_axis_dim;   // save reg_heap_axis_dim by rsp before using reg_inner_end
    Xbyak::Reg64 reg_heapify_i = reg_src;               // save reg_src by rsp before using reg_heapify_i
    Xbyak::Reg64 reg_heapify_valid = reg_heap_seq_idx;  // save reg_heap_seq_idx by rsp before using reg_heapify_valid
    Xbyak::Reg64 reg_heapify_tmp = reg_params;          // save reg_params by rsp before using reg_heapify_tmp

    // *** for bubble sort only ***
    Xbyak::Reg64 reg_bubble_seq_idx = reg_table;
    Xbyak::Reg64 reg_bubble_block_idx = reg_prc;
    Xbyak::Reg64 reg_bubble_axis_dim = reg_prc_idx;
    Xbyak::Reg64 reg_block_l = reg_bubble_block_idx;  // save reg_bubble_block_idx by rsp before using reg_l
    Xbyak::Reg64 reg_block_r = reg_bubble_axis_dim;   // save reg_bubble_axis_dim by rsp before using reg_r
    Xbyak::Reg64 reg_seq_l = reg_load_table;          // blocked layout on channel
    Xbyak::Reg64 reg_seq_r = reg_prc;                 // blocked layout on channel
    Xbyak::Reg64 reg_offset_l = reg_i;                // save reg_i by rsp before using reg_offset_l
    Xbyak::Reg64 reg_offset_r = reg_prc_idx;          // save reg_prc_idx by rsp before using reg_offset_r
    Xbyak::Reg64 reg_bubble_block_top_k = reg_bubble_seq_idx;
    Xbyak::Reg64 reg_bubble_block_k_sub_1 = reg_bubble_block_top_k;
    Xbyak::Reg64 reg_bubble_seq_top_k = reg_load_store_mask;
    Xbyak::Reg64 reg_bubble_seq_k_sub_1 = reg_bubble_seq_top_k;
    Xbyak::Reg64 reg_block_sort_stride = reg_aux;                     // by vector
    Xbyak::Reg64 reg_block_sort_stride_byte = reg_block_sort_stride;  // by vector
    Xbyak::Reg64 reg_seq_tmp = reg_seq_l;                             // blocked layout on channel
    Xbyak::Reg64 reg_seq_sort_stride = reg_work_amount;               // blocked layout on channel
    Xbyak::Reg64 reg_blk_stride =
        reg_seq_sort_stride;  // blocked layout on channel, denotes reg_seq_sort_stride * jcp_.blk_size
    Xbyak::Reg64 reg_sub_idx = reg_bubble_block_idx;  // blocked layout on channel
    // ========================================================================================================================

    Vmm vmm_zero = Vmm(0);  // vmm_zero represents Vmm(0) when isa is avx512_core, otherwise vmm_mask represents Vmm(0)

    const Xbyak::Opmask k_mask = Xbyak::Opmask(1);
    const int vector_step = vlen / sizeof(float);
    const int tail_step = jcp_.work_amount % vector_step;

    int blk_stride =
        0;  // stride of channel blocks at the same space coordinate, only used in blocked layout with topk on channel
    unsigned char cmp_flg;
    unsigned char heap_cmp_flg;

    Xbyak::Label l_table;

    std::unordered_map<size_t, std::unique_ptr<ov::intel_cpu::jit_emitter>> emitters;

    std::vector<size_t> store_pool_gpr_idxs;
    std::vector<size_t> load_pool_gpr_idxs;
    std::vector<size_t> store_pool_vec_idxs;

    void emit_emitters_data();

    inline void load(Xbyak::Reg64 reg_src, Vmm vmm_src, const int elt_num, const int offset = 0);

    inline void load_i32(Xbyak::Reg64 reg_src, Vmm vmm_src, const int elt_num, const int offset = 0);

    inline void store(Vmm vmm_dst, Xbyak::Reg64 reg_dst, const int elt_num, const int offset = 0);

    inline void store_i32(Vmm vmm_dst, Xbyak::Reg64 reg_dst, const int elt_num, const int offset = 0);

    inline void emit_load(Xbyak::Reg64 reg_src,
                          Vmm vmm_src,
                          ov::element::Type src_prc,
                          ov::element::Type dst_prc,
                          const int elt_num,
                          const int offset = 0);

    inline void emit_store(Vmm vmm_dst,
                           Xbyak::Reg64 reg_dst,
                           ov::element::Type src_prc,
                           ov::element::Type dst_prc,
                           const int elt_num,
                           const int offset = 0);

    inline void topk_loop();

    inline void topk_bitonic_vector();

    inline void topk_bitonic(int elt_num);

    // src memory layout: (N) * (CB * H * W * blk_size)
    // prc memory layout: (C) * (N * H * W)
    // topk_bitonic_BLK_on_channel: sort (C) * (N * H * W / blk_size * blk_size) elements
    //                              sort (C) * (N * H * W % blk_size) elements in the rear
    inline void topk_bitonic_BLK_on_channel();

    inline void bitonic_sort_vector(int elt_num, bool cmp_val = true);

    inline void bitonic_BLK_on_channel_load(int elt_num);

    inline void bitonic_BLK_on_channel_store(int elt_num);

    inline void bitonic_get_addr(Xbyak::Reg64 reg_base, int data_size, int offset = 0);

    inline void bitonic_swap_vector(int elt_num, bool cmp_val = true);

    inline void topk_heap_sorting();

    inline void topk_heap_load(Xbyak::Reg64& reg_end, int s);

    inline void topk_heap_extract(bool cmp_val = true);

    inline void heapify_sub_tree(const Xbyak::Reg64& reg_idx, const Xbyak::Reg64& reg_valid, bool cmp_val = true);

    inline bool is_valid_isa(dnnl::impl::cpu::x64::cpu_isa_t cpu_isa);

    inline void uni_vpcmpgtd(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2, const Xbyak::Operand& op);

    inline void uni_vpcmpgtd(const Xbyak::Ymm& x1, const Xbyak::Ymm& x2, const Xbyak::Operand& op);

    inline void compare_node_xmm(Xbyak::Xmm xmm_val_a,
                                 Xbyak::Xmm xmm_idx_a,
                                 Xbyak::Xmm xmm_val_b,
                                 Xbyak::Xmm xmm_idx_b,
                                 Xbyak::Xmm mask,
                                 unsigned char val_cmp_flg,
                                 unsigned char idx_cmp_flg,
                                 bool cmp_val);

    inline void heap_cmp_node(Xbyak::Xmm xmm_val_a, Xbyak::Xmm xmm_idx_a, Xbyak::Xmm xmm_val_b, Xbyak::Xmm xmm_idx_b, bool cmp_val = true);

    // n: node, c: child
    inline void heap_swap_node(Xbyak::Xmm xmm_val_n, Xbyak::Xmm xmm_idx_n, Xbyak::Xmm xmm_val_c, Xbyak::Xmm xmm_idx_c);

    inline void heap_swap_root(const Xbyak::Reg64& reg_idx);

    inline void topk_bubble_vector();

    inline void reg_add(const Xbyak::Reg64& reg_sum, const Xbyak::Reg64& reg_a, const Xbyak::Reg64& reg_b);

    inline void query_table_by_reg_idx(const Xbyak::Reg64& reg_table,
                                       const Xbyak::Reg64& reg_idx,
                                       int offset,
                                       size_t size);

    inline void table_to_vmm(Vmm vmm_src,
                             const Xbyak::Reg64& reg_table,
                             const Xbyak::Reg64& reg_idx,
                             int offset,
                             size_t size);

    inline void table_to_xmm(Xbyak::Xmm xmm_src,
                             const Xbyak::Reg64& reg_table,
                             const Xbyak::Reg64& reg_idx,
                             int offset,
                             size_t size);

    inline void get_addr_by_reg_idx(const Xbyak::Reg& reg_out,
                                    const Xbyak::Reg& reg_base,
                                    const Xbyak::Reg64& reg_in,
                                    int value);

    inline void get_addr_by_reg_idx(const Xbyak::Reg& reg_out,
                                    const Xbyak::Reg& reg_base,
                                    const Xbyak::Reg64& reg_in,
                                    int value,
                                    const Xbyak::Reg64& reg_value);

    inline void get_addr_by_reg_idx(const Xbyak::Reg& reg_out,
                                    const Xbyak::Reg& reg_base,
                                    const Xbyak::Reg64& reg_in,
                                    const Xbyak::Reg64& reg_value);

    inline void reg_mul_add(const Xbyak::Reg& reg_out, const Xbyak::Reg64& reg_in, int mul_val, int add_val);

    inline void reg_mul_add(const Xbyak::Reg& reg_out,
                            const Xbyak::Reg& reg_tmp,
                            const Xbyak::Reg64& reg_in,
                            int mul_val);

    inline void reg_mul_add(const Xbyak::Reg& reg_out, int mul_val, const Xbyak::Reg64& reg_base);

    inline void reg_sub_shr(const Xbyak::Reg& reg_out, const Xbyak::Reg64& reg_in, int sub_val, int shr_val);

    inline void reg_sub_mul(const Xbyak::Reg& reg_out, const Xbyak::Reg64& reg_in, int sub_val, int mul_val);

    inline void reg_shl(const Xbyak::Reg& reg_out, int rate);

    inline void reg_shr(const Xbyak::Reg& reg_out, int rate);

    inline void reg_div_blk_size(const Xbyak::Reg& reg_out, const Xbyak::Reg64& reg_in, int blk_size);

    inline void reg_mod_blk_size(const Xbyak::Reg& reg_out, const Xbyak::Reg64& reg_in, int blk_size);

    inline void reg_calc_offset_by_channel_idx(const Xbyak::Reg& reg_out,
                                               const Xbyak::Reg64& reg_stride,
                                               const Xbyak::Reg64& reg_channel_idx,
                                               int blk_size);

    inline void topk_bubble(int elt_num);

    inline void topk_bubble_vector_sort(int elt_num, bool cmp_val = true);

    inline void topk_bubble_inplace(int elt_num);

    inline void topk_bubble_horiz();

    // dst: xmm_val(0) and xmm_idx(0)
    // aux: xmm_val(2/3/4) and xmm_idx(2/3/4)
    inline void horiz_process();

    // dst: xmm_val(0) and xmm_idx(0)
    // aux: xmm_val(3) and xmm_idx(3)
    inline void horize_top1();

    inline void topk_bubble_BLK_on_channel_verti();

    inline void topk_bubble_BLK_on_channel();

    inline void topk_bubble_BLK_on_channel_sort(bool cmp_val = true);

    inline void topk_bubble_BLK_on_channel_inplace();

    inline void bubble_swap_vector(const Xbyak::Reg64& reg_l,
                                   const Xbyak::Reg64& reg_r,
                                   int elt_num,
                                   bool cmp_val = true);

    inline void swap_vector(Vmm vmm_val_a, Vmm vmm_idx_a, Vmm vmm_val_b, Vmm vmm_idx_b, bool cmp_val = true);

    inline void bubble_swap_by_index(const Xbyak::Reg64& reg_l, const Xbyak::Reg64& reg_r, bool cmp_val = true);

    inline void bubble_swap_xmm(Xbyak::Xmm xmm_val_a, Xbyak::Xmm xmm_idx_a, Xbyak::Xmm xmm_val_b, Xbyak::Xmm xmm_idx_b, bool cmp_val = true);

    inline void load_scalar(Xbyak::Xmm xmm_src, const Xbyak::Address& op, dnnl::memory::data_type src_dt, bool cvt_dt = false);

    inline void store_scalar(const Xbyak::Address& op, Xbyak::Xmm xmm_dst, dnnl::memory::data_type dst_dt, bool cvt_dt = false);

    void prepare_idx_table();

    void test();
};

}  // namespace kernel
}  // namespace intel_cpu
}  // namespace ov