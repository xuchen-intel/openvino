// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

enum ReduceLayoutType {
    reduce_ncsp,
    reduce_nspc,
    reduce_blocked
};

struct jit_reduce_config_params {
    ReduceLayoutType layout;
    Algorithm reduce_mode;
    mkldnn::memory::data_type src_dt;
    mkldnn::memory::data_type dst_dt;
    int src_data_size;
    int dst_data_size;
};

struct jit_reduce_call_args {
    const void *src;
    const int *idx;
    void *dst;
    size_t work_amount;
    size_t work_batch;
    size_t reduce_w = 2;    // only used in planar layout  [1: reduce width dimension]   [0: reduce other dimension] [other value: N/A]
    size_t reduce_stride;   // only used in planar layout while reducing dimensions except for width
};

struct jit_reduce_post_call_args {
    const void *src;
    void *dst;
    size_t work_amount;
    size_t reduce_c = 2;    // only used in blocked layout [1: reduce channel dimension] [0: reduce other dimension] [other value: N/A]
    size_t oc_off;          // offset in byte along channel on output tensor
    size_t channel_size;    // only for post ops fusion of nspc layout
    const float *divisor;   // mean = sum / divisor
};

struct jit_uni_reduce_kernel {
    void (*ker_)(const jit_reduce_call_args *);

    void operator()(const jit_reduce_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_reduce_kernel(jit_reduce_config_params jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_reduce_kernel() {}

    virtual void create_ker() = 0;

    jit_reduce_config_params jcp_;
};

struct jit_uni_reduce_post_kernel {
    void (*ker_)(const jit_reduce_post_call_args *);

    void operator()(const jit_reduce_post_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_reduce_post_kernel(jit_reduce_config_params jcp, const mkldnn_primitive_attr &attr) : ker_(nullptr), jcp_(jcp), attr_(attr) {}
    virtual ~jit_uni_reduce_post_kernel() {}

    virtual void create_ker() = 0;

    jit_reduce_config_params jcp_;
    const mkldnn_primitive_attr &attr_;
};

class MKLDNNReduceNode : public MKLDNNNode {
public:
    MKLDNNReduceNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(mkldnn::stream strm) override;
    bool canFuse(const MKLDNNNodePtr& node) const override;
    bool canBeInPlace() const override {
        return false;
    }

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    void reduce_type(const uint8_t *in_ptr, uint8_t *out_ptr, size_t dst_size);
    void reduce_PLN(const uint8_t *in_ptr, uint8_t *out_ptr);
    void reduce_BLK(const uint8_t *in_ptr, uint8_t *out_ptr);
    void reduce_BLK_concern_padding(const uint8_t *in_ptr, uint8_t *out_ptr);
    inline void reduce_kernel_process(const uint8_t *in_p, uint8_t *out_p, size_t work_amount,
                                      size_t reduce_w = 2, size_t work_batch = 1, const int *tab_idx = NULL);
    inline void reduce_kernel_post_process(uint8_t *out_ptr);
    inline void init_dst_data(uint8_t *out_ptr, size_t dst_size);
    inline void create_working_memory();
    inline void calc_process_dst_dims();
    inline void set_reduce_dim_flags();
    inline void reduce_ref(const float *in_ptr, float *out_ptr);
    void reduce_ref_process(const float *in_ptr, float *out_ptr, float init_value, std::function<float(float, float)> func);
    inline void reduce_ref_map(float *out_ptr, size_t work_amount_dst, size_t reduced_dims_work_amount);
    void nspc2ncsp(uint8_t *proc_ptr, uint8_t *out_ptr);
    void blocked2ncsp(uint8_t *proc_ptr, uint8_t *out_ptr);
    void setPostOps(mkldnn::primitive_attr &attr, bool initWeights = false);
    void setJITBeyond5D();
    void update_src_dims();
    bool canApplyJIT(const InferenceEngine::Precision &input_prec, const InferenceEngine::Precision &output_prec) const;

    size_t blk_size;
    size_t dst_size;
    static const size_t REDUCE_DATA = 0;
    static const size_t REDUCE_INDEXES = 1;
    bool jit_beyond_5D = false;
    bool jit_mode = true;
    bool keep_dims = true;
    bool is_hybrid_layout = false;
    bool ReduceN, ReduceC, ReduceD, ReduceH, ReduceW;
    size_t IB, IC, ID, IH, IW;
    size_t OB, OC, OD, OH, OW;
    size_t src_data_size, dst_data_size;
    size_t reduce_stride;
    ReduceLayoutType layout;
    InferenceEngine::Precision input_prec, output_prec;
    InferenceEngine::SizeVector src_dims;
    InferenceEngine::SizeVector process_dst_dims;
    InferenceEngine::SizeVector axes_for_reduction;
    std::vector<int> raw_axes;

    mkldnn::primitive_attr attr;

    std::shared_ptr<mkldnn::memory> prc_mem;

    std::shared_ptr<jit_uni_reduce_kernel> reduce_kernel;
    std::shared_ptr<jit_uni_reduce_post_kernel> reduce_post_kernel;

    static const std::map<const ngraph::DiscreteTypeInfo, std::function<void(const std::shared_ptr<ngraph::Node>& op, MKLDNNReduceNode& node)>> initializers;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin

