// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernels/x64/topk_kernel.hpp"

#include <memory>
#include <string>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class TopK : public Node {
public:
    TopK(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);
    ~TopK() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    bool needShapeInfer() const override;
    bool needPrepareParams() const override;
    void prepareParams() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    bool canBeInPlace() const override {
        return false;
    }

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    void topk_process(const uint8_t* in_ptr, uint8_t* out_ptr, uint8_t* dst_idx);
    void topk_ref(const float* in_ptr, float* out_ptr, int32_t* dst_idx);
    inline void topk_kernel_process(const uint8_t* in_p,
                                    uint8_t* out_p,
                                    uint8_t* src_idx,
                                    uint8_t* process_p,
                                    uint8_t* process_idx_p,
                                    size_t work_amount);
    inline static int count(const VectorDims& dims, size_t start_ind, size_t end_ind);
    inline static int count(const VectorDims& dims, size_t start_ind = 0);
    inline void bitonic_push_idx(int p, int n, std::vector<int>& vec, int& cnt, bool cmp_val = true);
    void calc_bitonic_idx(size_t n, int& cnt, bool cmp_val);
    void calc_dims_size(const VectorDims& layout_dims);
    void topk_ref_process(const float* src_data,
                          float* dst_data,
                          int32_t* dst_idx,
                          const VectorDims& in_dims,
                          std::function<bool(float, float)> compare) const;
    void preset_params();
    void prepare_original_idx();

    bool topk_innermost = false;
    bool jit_mode = false;
    bool sort_index = false;
    bool stable = false;
    bool mode_max = false;
    int axis = 0;
    static const size_t TOPK_DATA = 0;
    static const size_t TOPK_K = 1;
    static const size_t TOPK_INDEX = 1;
    size_t O = 0, A = 0, I = 0;
    size_t blk_size = 0;
    size_t data_size = 0;
    size_t axis_dim = 0;
    int top_k = 0;
    int dim = 0, before_num = 0;
    bool bubble_inplace = false;
    bool preset_params_done = false;

    VectorDims src_dims, dst_dims;
    kernel::TopKLayoutType layout = kernel::TopKLayoutType::topk_ncsp;
    kernel::TopKAlgorithm algorithm = kernel::TopKAlgorithm::topk_bubble_sort;

    std::vector<int> vec_bitonic_idx;
    std::vector<int> vec_bitonic_k_idx;

    std::vector<int> vec_idx_seq;
    std::vector<int> vec_idx_block;

    std::vector<uint8_t> vec_process_ptr;
    std::vector<uint8_t> vec_process_idx_ptr;

    std::shared_ptr<kernel::jit_uni_topk_kernel> topk_kernel = nullptr;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
