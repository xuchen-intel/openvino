// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "topk.h"

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "common/cpu_memcpy.h"
#include "cpu/x64/jit_generator.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/op/topk.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/ngraph_utils.hpp"

using namespace dnnl;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

namespace ov::intel_cpu::node {

bool TopK::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                    ov::op::v1::TopK::get_type_info_static(),
                    ov::op::v3::TopK::get_type_info_static(),
                    ov::op::v11::TopK::get_type_info_static())) {
            errorMessage = "Node is not an instance of the TopK from the operation sets v1, v3 or v11";
            return false;
        }

        auto topKOp = ov::as_type_ptr<const ov::op::util::TopKBase>(op);
        if (!isDynamicNgraphNode(op)) {
            auto topKConst = ov::as_type_ptr<const ov::op::v0::Constant>(topKOp->get_input_node_shared_ptr(TOPK_K));
            if (!topKConst) {
                errorMessage = "Second tensor is not constant in static shape mode";
                return false;
            }
        }

        if (topKOp->get_mode() != ov::op::TopKMode::MAX && topKOp->get_mode() != ov::op::TopKMode::MIN) {
            errorMessage = "Unsupported mode.";
            return false;
        }
        if (!one_of(topKOp->get_sort_type(),
                    ov::op::TopKSortType::NONE,
                    ov::op::TopKSortType::SORT_VALUES,
                    ov::op::TopKSortType::SORT_INDICES)) {
            errorMessage = "Unsupported sort type.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

TopK::TopK(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        auto topKOp = ov::as_type_ptr<const ov::op::util::TopKBase>(op);

        auto in_dims = topKOp->get_input_partial_shape(TOPK_DATA);
        auto out_dims = topKOp->get_output_partial_shape(TOPK_DATA);
        auto out_idx_dims = topKOp->get_output_partial_shape(TOPK_INDEX);
        auto in_dims_size = in_dims.size();

        if (!isDynamicNgraphNode(op)) {
            auto topKConst = ov::as_type_ptr<const ov::op::v0::Constant>(topKOp->get_input_node_shared_ptr(TOPK_K));
            if (!topKConst) {
                THROW_CPU_NODE_ERR("gets non-constant second tensor in static shape mode!");
            }
        }

        axis = topKOp->get_axis();
        mode_max = topKOp->get_mode() == ov::op::TopKMode::MAX;
        sort_index = topKOp->get_sort_type() == ov::op::TopKSortType::SORT_INDICES;

        stable = false;
        if (!sort_index) {
            const auto topKOpV11 = ov::as_type_ptr<const ov::op::v11::TopK>(op);
            if (topKOpV11) {
                stable = topKOpV11->get_stable();
            }
        }

        top_k = 0;
        preset_params_done = false;
        vec_idx_seq.clear();
        vec_idx_block.clear();

        if (inputShapes.size() != 2 || outputShapes.size() < 2) {
            THROW_CPU_NODE_ERR("gets incorrect number of input/output edges!");
        }

        if (getInputShapeAtPort(TOPK_DATA).getRank() != getOutputShapeAtPort(TOPK_DATA).getRank()) {
            THROW_CPU_NODE_ERR("gets incorrect number of input/output dimensions!");
        }

        if (getInputShapeAtPort(TOPK_K).getRank() != 1) {
            THROW_CPU_NODE_ERR("gets incorrect index vector dimension! Index vector should be 1 dimension.");
        }

        if (out_dims != out_idx_dims) {
            THROW_CPU_NODE_ERR("gets incorrect output tensor dimension sizes!");
        }

        if (axis < 0) {
            axis += in_dims_size;
        }
        if (axis < 0 || axis >= static_cast<int>(in_dims_size)) {
            THROW_CPU_NODE_ERR("gets incorrect input parameters dimensions and axis number!");
        }
    } else {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

void TopK::getSupportedDescriptors() {}

void TopK::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    impl_desc_type impl_type;
    if (mayiuse(cpu::x64::avx512_core)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (mayiuse(cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    } else if (mayiuse(cpu::x64::sse41)) {
        impl_type = impl_desc_type::jit_sse42;
    } else {
        impl_type = impl_desc_type::ref;
    }

#if defined(OPENVINO_ARCH_X86_64)
    jit_mode = mayiuse(cpu::x64::sse41);
#else
    jit_mode = false;
#endif

    static const ov::element::Type supportedPrecision[] = {ov::element::f32,
                                                           ov::element::bf16,
                                                           ov::element::i32,
                                                           ov::element::i8,
                                                           ov::element::u8};

    ov::element::Type dataPrecision = getOriginalOutputPrecisionAtPort(TOPK_DATA);
    bool precisionSupported = std::find(std::begin(supportedPrecision), std::end(supportedPrecision), dataPrecision) !=
                              std::end(supportedPrecision);
    precisionSupported = (dataPrecision == ov::element::bf16 && !mayiuse(avx512_core)) ? false : precisionSupported;
    if (!precisionSupported) {
        if (dataPrecision.is_real()) {
            dataPrecision = ov::element::f32;
        } else {
            dataPrecision = ov::element::i32;
        }
    }

    std::vector<std::pair<LayoutType, LayoutType>> dataFomats {
        {LayoutType::ncsp, LayoutType::ncsp},
#if defined(OPENVINO_ARCH_X86_64)
            {LayoutType::nspc, LayoutType::nspc}, {LayoutType::nCsp16c, LayoutType::nCsp16c}, {
            LayoutType::nCsp8c, LayoutType::nCsp8c
        }
#endif
    };

    for (const auto& df : dataFomats) {
        addSupportedPrimDesc({{df.first, dataPrecision}, {LayoutType::ncsp, ov::element::i32}},
                             {{df.second, dataPrecision}, {df.second, ov::element::i32}},
                             impl_type);
    }
}

bool TopK::needShapeInfer() const {
    const int src_k = getSrcDataAtPortAs<int>(TOPK_K)[0];
    return inputShapesModified() || src_k != top_k;
}

bool TopK::needPrepareParams() const {
    const int src_k = getSrcDataAtPortAs<int>(TOPK_K)[0];
    return inputShapesModified() || top_k != src_k;
}

void TopK::preset_params() {
    auto selectedPD = getSelectedPrimitiveDescriptor();
    auto data_type = DnnlExtensionUtils::ElementTypeToDataType(
        selectedPD->getConfig().inConfs[TOPK_DATA].getMemDesc()->getPrecision());
    data_size = DnnlExtensionUtils::sizeOfDataType(data_type);

    topk_innermost = (layout == kernel::TopKLayoutType::topk_ncsp &&
                      axis == static_cast<int>(getOutputShapeAtPort(TOPK_DATA).getRank() - 1)) ||
                     ((layout == kernel::TopKLayoutType::topk_nspc || layout == kernel::TopKLayoutType::topk_blocked) && axis == 1);

    if (mayiuse(cpu::x64::avx512_core)) {
        blk_size = 16;
    } else if (mayiuse(cpu::x64::sse41)) {
        blk_size = 8;
    }

    bool can_use_heap_sort =
        (layout == kernel::TopKLayoutType::topk_ncsp || layout == kernel::TopKLayoutType::topk_nspc) && topk_innermost;
    bool use_bubble_sort = stable || !can_use_heap_sort;
    if (isDynamicNode()) {
        if (use_bubble_sort) {
            algorithm = kernel::TopKAlgorithm::topk_bubble_sort;
            bubble_inplace = false;
        } else {
            algorithm = kernel::TopKAlgorithm::topk_heap_sort;
        }
    }
}

void TopK::prepareParams() {
    auto dstMemPtr = getDstMemoryAtPort(TOPK_DATA);
    auto srcMemPtr = getSrcMemoryAtPort(TOPK_DATA);
    if (!dstMemPtr || !dstMemPtr->isDefined()) {
        THROW_CPU_NODE_ERR("has undefined destination memory.");
    }
    if (!srcMemPtr || !srcMemPtr->isDefined()) {
        THROW_CPU_NODE_ERR("has undefined input memory.");
    }
    if (getSelectedPrimitiveDescriptor() == nullptr) {
        THROW_CPU_NODE_ERR("has nullable preferable primitive descriptor");
    }

    src_dims = srcMemPtr->getDesc().getShape().getDims();
    dst_dims = dstMemPtr->getDesc().getShape().getDims();

    if (isDynamicNode()) {
        const int src_k = getSrcDataAtPortAs<int>(TOPK_K)[0];
        if (static_cast<size_t>(src_k) > src_dims[axis]) {
            THROW_CPU_NODE_ERR("gets top_k out of range!");
        }
        if (top_k != src_k) {
            top_k = src_k;
        }
    } else {
        top_k = getSrcDataAtPortAs<int>(TOPK_K)[0];
    }

    if (jit_mode) {
        if (!preset_params_done) {
            preset_params();
            preset_params_done = true;
        }

        auto layout_dims = dstMemPtr->getDescWithType<BlockedMemoryDesc>()->getBlockDims();
        calc_dims_size(layout_dims);

        axis_dim = src_dims[axis];

        // [case 1]: if 2 * (top_k + 1) + 2 <= count_xmm, thus top_k is small enough that the vector registers are
        // sufficient
        //           to keep all necessary data for sorting, no need to load and store frequently, use inplace bubble
        //           sort; (horizotal sorting cases not included)
        // [case 2]: if stable sorting is required, bubble sort(topk_bubble_vector/topk_bubble_BLK_on_channel_verti)
        // will be
        //           applied currently, because among the implemented sorting algorithms, these bubble sort
        //           implementations are the only stable ones;
        // [case 3]: only when topk is imposed on innermost dimsension of planar(ncsp/nspc) layout, should heap sort be
        // used; [case 4]: by default, use bitonic sort when alg_cost_bitonic < alg_cost_bubble, otherwise use bubble
        // sort.
        //           alg_cost_bitonic = (N / 4) * logN * (logN + 1)
        //           alg_cost_bubble = K * (K - 1) / 2 + (N - K) * K
        //           where, N = axis_dim, K = topk_k
        //           the above two alg_costs are not the exact implementation costs, yet it's proper to use them to
        //           decide which algorithm should be used for specific N and K.
        if (!isDynamicNode()) {
            const size_t count_xmm = 16;  // only 16 vector registers are valid in sse instructions even for avx512_core
            if (static_cast<size_t>(top_k) <= count_xmm / 2 - 2) {
                algorithm = kernel::TopKAlgorithm::topk_bubble_sort;
                bubble_inplace = topk_innermost && top_k == 1 ? false : true;
            } else if (stable) {
                algorithm = kernel::TopKAlgorithm::topk_bubble_sort;
                bubble_inplace = false;
            } else if ((layout == kernel::TopKLayoutType::topk_ncsp || layout == kernel::TopKLayoutType::topk_nspc) && topk_innermost) {
                algorithm = kernel::TopKAlgorithm::topk_heap_sort;
            } else {
                auto log_axis_dim = log2(axis_dim);
                auto alg_cost_bitonic = static_cast<size_t>((axis_dim / 4.0f) * log_axis_dim * (log_axis_dim + 1));
                size_t alg_cost_bubble = top_k * (top_k - 1) / 2 + (axis_dim - top_k) * top_k;
                if (alg_cost_bitonic < alg_cost_bubble) {
                    algorithm = kernel::TopKAlgorithm::topk_bitonic_sort;
                } else {
                    algorithm = kernel::TopKAlgorithm::topk_bubble_sort;
                    bubble_inplace = false;
                }
            }
        }

        prepare_original_idx();
    } else {  // reference mode
        int j;
        for (j = src_dims.size() - 1; j >= 0; j--) {
            if (src_dims[j] != 1) {
                break;
            }
        }
        dim = static_cast<int>(src_dims[axis]);
        before_num = count(src_dims, 0, axis);
    }
}

void TopK::createPrimitive() {
    auto srcMemPtr = getSrcMemoryAtPort(TOPK_DATA);
    if (srcMemPtr->getDesc().hasLayoutType(LayoutType::ncsp)) {
        layout = kernel::TopKLayoutType::topk_ncsp;
    } else if (srcMemPtr->getDesc().hasLayoutType(LayoutType::nspc)) {
        layout = kernel::TopKLayoutType::topk_nspc;
    } else {
        layout = kernel::TopKLayoutType::topk_blocked;
    }

    if (!isDynamicNode() && isExecutable()) {
        if (needPrepareParams()) {
            prepareParams();
        }
        updateLastInputDims();
    }

    if (jit_mode) {
        if (!preset_params_done) {
            preset_params();
            preset_params_done = true;
        }

        // Shape related config params will only be used for static shape sorting algorithms.
        // Such params are useless for dynamic shapes, instead their jit_topk_call_args counterparts
        // will be used. These params are: top_k, axis_dim, sort_stride, work_amount
        auto jcp = kernel::jit_topk_config_params();
        auto selectedPD = getSelectedPrimitiveDescriptor();
        jcp.precision = selectedPD->getConfig().inConfs[TOPK_DATA].getMemDesc()->getPrecision();
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
        jcp.stable = stable;
        jcp.sort_stride = static_cast<int>(I);
        jcp.work_amount = static_cast<int>(I);
        jcp.bitonic_idx_cnt = 0;
        jcp.bitonic_k_idx_cnt = 0;

        if (algorithm == kernel::TopKAlgorithm::topk_bitonic_sort) {
            size_t src_count = srcMemPtr->getDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();
            vec_process_ptr.resize(src_count * data_size);
            vec_process_idx_ptr.resize(src_count * sizeof(int32_t));

            calc_bitonic_idx(axis_dim, jcp.bitonic_idx_cnt, true);
            if (sort_index) {
                calc_bitonic_idx(top_k, jcp.bitonic_k_idx_cnt, false);
            }
        }
#if defined(OPENVINO_ARCH_X86_64)
        if (mayiuse(cpu::x64::avx512_core)) {
            topk_kernel = std::make_shared<kernel::jit_uni_topk_kernel_f32<cpu::x64::avx512_core>>(jcp);
        } else if (mayiuse(cpu::x64::avx2)) {
            topk_kernel = std::make_shared<kernel::jit_uni_topk_kernel_f32<cpu::x64::avx2>>(jcp);
        } else if (mayiuse(cpu::x64::sse41)) {
            topk_kernel = std::make_shared<kernel::jit_uni_topk_kernel_f32<cpu::x64::sse41>>(jcp);
        }

        if (topk_kernel) {
            topk_kernel->create_ker();
        }
#endif
    }
}

void TopK::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void TopK::execute([[maybe_unused]] const dnnl::stream& strm) {
    auto srcMemPtr = getSrcMemoryAtPort(TOPK_DATA);
    auto dstMemPtr = getDstMemoryAtPort(TOPK_DATA);
    auto dstIndexesMemPtr = getDstMemoryAtPort(TOPK_INDEX);

    const auto* src_data = srcMemPtr->getDataAs<const uint8_t>();
    auto* dst_data = dstMemPtr->getDataAs<uint8_t>();
    auto* dst_idx = dstIndexesMemPtr->getDataAs<uint8_t>();

    if (jit_mode) {
        topk_process(src_data, dst_data, dst_idx);
    } else {
        if (layout == kernel::TopKLayoutType::topk_ncsp) {
            auto in_ptr = reinterpret_cast<const float*>(src_data);
            auto out_ptr = reinterpret_cast<float*>(dst_data);
            auto out_idx_ptr = reinterpret_cast<int32_t*>(dst_idx);
            topk_ref(in_ptr, out_ptr, out_idx_ptr);
        } else {
            THROW_CPU_NODE_ERR("only support plain layout on machine w/o sse42.");
        }
    }
}

void TopK::topk_process(const uint8_t* in_ptr, uint8_t* out_ptr, uint8_t* out_idx_ptr) {
    uint8_t* process_ptr = vec_process_ptr.data();
    uint8_t* process_idx_ptr = vec_process_idx_ptr.data();

    // [blocked layout with topk on C]
    if (layout == kernel::TopKLayoutType::topk_blocked && topk_innermost) {
        size_t IA = div_up(src_dims[1], blk_size);
        size_t OA = div_up(dst_dims[1], blk_size);
        if (algorithm == kernel::TopKAlgorithm::topk_bubble_sort) {
            parallel_for2d(O, I, [&](size_t o, size_t i) {
                const uint8_t* in_ptr_a = in_ptr + (o * IA * I + i) * blk_size * data_size;
                uint8_t* out_ptr_a = out_ptr + (o * OA * I + i) * blk_size * data_size;
                uint8_t* out_idx_ptr_a = out_idx_ptr + (o * OA * I + i) * blk_size * sizeof(int32_t);
                size_t work_amount = 1;
                topk_kernel_process(in_ptr_a, out_ptr_a, out_idx_ptr_a, nullptr, nullptr, work_amount);
            });
        } else if (algorithm == kernel::TopKAlgorithm::topk_bitonic_sort) {
            parallel_for(O, [&](size_t o) {
                const uint8_t* in_ptr_a = in_ptr + o * IA * I * blk_size * data_size;
                uint8_t* process_ptr_a = process_ptr + o * IA * I * blk_size * data_size;
                uint8_t* process_idx_ptr_a = process_idx_ptr + o * IA * I * blk_size * sizeof(int32_t);
                uint8_t* out_ptr_a = out_ptr + o * OA * I * blk_size * data_size;
                uint8_t* out_idx_ptr_a = out_idx_ptr + o * OA * I * blk_size * sizeof(int32_t);
                size_t work_amount = I;
                topk_kernel_process(in_ptr_a, out_ptr_a, out_idx_ptr_a, process_ptr_a, process_idx_ptr_a, work_amount);
            });
        }
    } else {  // [planar layout] [blocked layout with topk on non-C]
        parallel_for2d(O, I / blk_size, [&](size_t o, size_t k) {
            const uint8_t* in_ptr_a = in_ptr + (o * A * I + k * blk_size) * data_size;
            uint8_t* process_ptr_a = process_ptr + (o * A * I + k * blk_size) * data_size;
            uint8_t* process_idx_ptr_a = process_idx_ptr + (o * A * I + k * blk_size) * sizeof(int32_t);
            uint8_t* out_ptr_a = out_ptr + (o * top_k * I + k * blk_size) * data_size;
            uint8_t* out_idx_ptr_a = out_idx_ptr + (o * top_k * I + k * blk_size) * sizeof(int32_t);
            size_t work_amount = blk_size;
            topk_kernel_process(in_ptr_a, out_ptr_a, out_idx_ptr_a, process_ptr_a, process_idx_ptr_a, work_amount);
        });

        size_t tail_start = I / blk_size * blk_size;
        size_t work_amount = I - tail_start;
        if (work_amount) {
            parallel_for(O, [&](size_t o) {
                const uint8_t* in_ptr_a = in_ptr + (o * A * I + tail_start) * data_size;
                uint8_t* process_ptr_a = process_ptr + (o * A * I + tail_start) * data_size;
                uint8_t* process_idx_ptr_a = process_idx_ptr + (o * A * I + tail_start) * sizeof(int32_t);
                uint8_t* out_ptr_a = out_ptr + (o * top_k * I + tail_start) * data_size;
                uint8_t* out_idx_ptr_a = out_idx_ptr + (o * top_k * I + tail_start) * sizeof(int32_t);
                topk_kernel_process(in_ptr_a, out_ptr_a, out_idx_ptr_a, process_ptr_a, process_idx_ptr_a, work_amount);
            });
        }
    }
}

inline void TopK::topk_kernel_process(const uint8_t* in_p,
                                      uint8_t* out_p,
                                      uint8_t* out_idx_p,
                                      uint8_t* process_p,
                                      uint8_t* process_idx_p,
                                      size_t work_amount) {
    auto arg = kernel::jit_topk_call_args();
    arg.src = static_cast<const void*>(in_p);
    arg.process = static_cast<void*>(process_p);
    arg.process_index = static_cast<void*>(process_idx_p);
    arg.dst = static_cast<void*>(out_p);
    arg.index = static_cast<void*>(out_idx_p);
    arg.work_amount = work_amount;
    arg.bitonic_idx_buf = vec_bitonic_idx.data();
    arg.bitonic_k_idx_buf = vec_bitonic_k_idx.data();
    arg.axis_dim = axis_dim;
    arg.top_k = static_cast<size_t>(top_k);
    arg.sort_stride = I;
    arg.idx_block_buf = vec_idx_block.data();
    arg.idx_seq_buf = vec_idx_seq.data();
    (*topk_kernel)(&arg);
}

inline void TopK::prepare_original_idx() {
    bool shape_agnostic_alg =
        algorithm == kernel::TopKAlgorithm::topk_heap_sort || (algorithm == kernel::TopKAlgorithm::topk_bubble_sort && !bubble_inplace);
    if (shape_agnostic_alg) {
        bool use_idx_seq = stable
                               ? topk_innermost && (layout == kernel::TopKLayoutType::topk_blocked || (top_k == 1 && !stable))
                               : topk_innermost;
        if (use_idx_seq) {
            if (vec_idx_seq.empty()) {
                vec_idx_seq.resize(axis_dim);
                std::iota(vec_idx_seq.begin(), vec_idx_seq.end(), 0);
            } else {
                size_t pre_size = vec_idx_seq.size();
                if (pre_size != axis_dim) {
                    vec_idx_seq.resize(axis_dim);
                    for (size_t i = pre_size; i < axis_dim; i++) {
                        vec_idx_seq[i] = i;
                    }
                }
            }
        } else {
            size_t blk_len = mayiuse(cpu::x64::avx2) ? blk_size : 4;
            if (vec_idx_block.empty()) {
                vec_idx_block.resize(axis_dim * blk_len);
                for (size_t i = 0; i < axis_dim; i++) {
                    for (size_t j = 0; j < blk_len; j++) {
                        vec_idx_block[i * blk_len + j] = i;
                    }
                }
            } else {
                size_t pre_size = vec_idx_block.size() / blk_len;
                if (pre_size != axis_dim) {
                    vec_idx_block.resize(axis_dim * blk_len);
                    for (size_t i = pre_size; i < axis_dim; i++) {
                        for (size_t j = 0; j < blk_len; j++) {
                            vec_idx_block[i * blk_len + j] = i;
                        }
                    }
                }
            }
        }
    }
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
inline void TopK::bitonic_push_idx(int p, int n, std::vector<int>& vec, int& cnt, bool cmp_val) {
    // memory stride of adjacent elements in sorting
    auto sort_stride = static_cast<int>(I);
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
        for (int sub_start = 0; (!cmp_val || (cmp_val && sub_start < n)) && sub_start < p; sub_start += sub_p) {
            int minor_p = sub_p >> 1;
            for (int j = 0; sub_start + j + minor_p < n && j < minor_p; j++) {
                vec[cnt++] = (sub_start + j) * sort_stride;
                vec[cnt++] = (sub_start + j + minor_p) * sort_stride;
            }
        }
    }
}

void TopK::calc_bitonic_idx(size_t n, int& cnt, bool cmp_val) {
    int m = n - 1;
    int log_p = 0;
    int p = 1;
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
void TopK::calc_dims_size(const VectorDims& layout_dims) {
    O = 1, I = 1;
    A = src_dims[axis];
    int layout_axis = axis;
    if (layout == kernel::TopKLayoutType::topk_nspc) {
        layout_axis = axis == 0 ? 0 : (axis == 1 ? static_cast<int>(layout_dims.size() - 1) : axis - 1);
    }

    for (int i = 0; i < layout_axis; i++) {
        O *= layout_dims[i];
    }
    for (size_t i = layout_axis + 1; i < layout_dims.size(); i++) {
        I *= layout_dims[i];
    }
    if (layout == kernel::TopKLayoutType::topk_blocked && topk_innermost) {
        I /= blk_size;
    }
}

void TopK::topk_ref(const float* in_ptr, float* out_ptr, int32_t* dst_idx) {
    if (mode_max) {
        topk_ref_process(in_ptr, out_ptr, dst_idx, src_dims, [](float x, float y) -> bool {
            return x > y;
        });
    } else {
        topk_ref_process(in_ptr, out_ptr, dst_idx, src_dims, [](float x, float y) -> bool {
            return x < y;
        });
    }
}

void TopK::topk_ref_process(const float* src_data,
                            float* dst_data,
                            int32_t* dst_idx,
                            const VectorDims& in_dims,
                            std::function<bool(float, float)> compare) const {
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
                if (compare(max_values[i3], max_values[i3 - 1])) {
                    swap_func(i3, i3 - 1);
                } else {
                    break;
                }
            }
            s_index += after_num;
        }
        if (sort_index) {
            for (int i2 = 0; i2 < top_k - 1; i2++) {
                for (int i3 = top_k - 1; i3 > i2; i3--) {
                    if (std::greater<>()(max_indexes[i3 - 1], max_indexes[i3])) {
                        swap_func(i3, i3 - 1);
                    }
                }
            }
        }
        if (dst_data) {
            for (int i2 = 0; i2 < top_k; i2++) {
                dst_data[i0 * top_k * after_num + i2 * after_num + i1] = max_values[i2];
            }
        }
        if (dst_idx) {
            for (int i2 = 0; i2 < top_k; i2++) {
                dst_idx[i0 * top_k * after_num + i2 * after_num + i1] = max_indexes[i2];
            }
        }
    });
}

inline int TopK::count(const VectorDims& dims, size_t start_ind, size_t end_ind) {
    size_t count = 1;
    for (size_t i = start_ind; i < end_ind; i++) {
        count *= dims[i];
    }
    return static_cast<int>(count);
}

inline int TopK::count(const VectorDims& dims, size_t start_ind) {
    return count(dims, start_ind, dims.size());
}

bool TopK::created() const {
    return getType() == Type::TopK;
}

}  // namespace ov::intel_cpu::node
