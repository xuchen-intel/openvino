// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

namespace ov::intel_cpu::node {

class RoPE : public Node {
public:
    RoPE(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    bool created() const override {
        return getType() == Type::RoPE;
    }
    bool needPrepareParams() const override {
        return false;
    };
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    struct Executor {
        virtual void execute(const dnnl::stream& strm,
                             const std::vector<MemoryPtr>& inputs,
                             const std::vector<MemoryPtr>& outputs) = 0;
        virtual ~Executor() = default;
    };
    template <typename T>
    struct RoPEExecutorRotateHalf;
    template <typename T>
    struct RoPEExecutorInterleaved;
    template <typename T>
    struct RoPEExecutorChatGLM;
    template <typename T>
    struct RoPEExecutorQwen;
    op::internal::RoPE::Config m_config;
    std::shared_ptr<Executor> m_executor;
};

}  // namespace ov::intel_cpu::node
