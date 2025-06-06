// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_executor.h"

#include <oneapi/dnnl/dnnl_types.h>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <unordered_map>

#include "dnnl_extension_utils.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
using namespace dnnl;

namespace ov::intel_cpu {

DnnlExecutorLegacy::DnnlExecutorLegacy(const dnnl::primitive_desc& pd) {
    execPrim = dnnl::primitive(pd);
    src_md = DnnlExtensionUtils::makeDescriptor(pd.src_desc());
    dst_md = DnnlExtensionUtils::makeDescriptor(pd.dst_desc());
    wghts_md = DnnlExtensionUtils::makeDescriptor(pd.weights_desc());
    scrch_md = DnnlExtensionUtils::makeDescriptor(pd.scratchpad_desc());
}

DnnlExecutorLegacy::IntermReorder::IntermReorder(const dnnl::memory::desc& descSrc,
                                                 const dnnl::memory::desc& descDst,
                                                 const dnnl::engine& engine)
    : m_descSrc(descSrc),
      m_descDst(descDst) {
    auto reorderPd = dnnl::reorder::primitive_desc(engine, descSrc, engine, descDst);
    m_reorder = dnnl::reorder(reorderPd);
}

void DnnlExecutorLegacy::IntermReorder::exec(dnnl::memory& memSrc, dnnl::memory& memDst, const dnnl::stream& strm) {
    m_reorder.execute(strm, memSrc, memDst);
}

void DnnlExecutorLegacy::exec(const std::unordered_map<int, dnnl::memory>& primArgs, const dnnl::stream& strm) {
    if (inputReorders.empty() && outputReorders.empty()) {
        execPrim.execute(strm, primArgs);
    } else {
        reorder_exec(primArgs, strm);
    }
}

void DnnlExecutorLegacy::reorder_exec(std::unordered_map<int, dnnl::memory> primArgs, const dnnl::stream& strm) {
    for (auto& inReorder : inputReorders) {
        if (primArgs.count(inReorder.first)) {
            dnnl::memory memDst(inReorder.second.getDstDesc(), strm.get_engine());
            inReorder.second.exec(primArgs[inReorder.first], memDst, strm);
            primArgs[inReorder.first] = memDst;
        } else {
            OPENVINO_THROW("DnnlExecutor has reorder for input ", inReorder.first, ", but doesn't have source memory");
        }
    }
    std::unordered_map<int, dnnl::memory> outputMem;
    for (auto& outReorder : outputReorders) {
        if (primArgs.count(outReorder.first)) {
            dnnl::memory memSrc(outReorder.second.getSrcDesc(), strm.get_engine());
            outputMem[outReorder.first] = primArgs[outReorder.first];
            primArgs[outReorder.first] = memSrc;
        } else {
            OPENVINO_THROW("DnnlExecutor has reorder for output ",
                           outReorder.first,
                           ", but doesn't have destination memory");
        }
    }
    execPrim.execute(strm, primArgs);
    for (auto& outReorder : outputReorders) {
        outReorder.second.exec(primArgs[outReorder.first], outputMem[outReorder.first], strm);
    }
}

bool DnnlExecutorLegacy::needReordering() const {
    return !inputReorders.empty() || !outputReorders.empty();
}

dnnl::primitive DnnlExecutorLegacy::getExecPrim() const {
    return execPrim;
}

const_dnnl_primitive_desc_t DnnlExecutorLegacy::getPrimitiveDesc() const {
    return execPrim.get_primitive_desc();
}

impl_desc_type DnnlExecutorLegacy::getImplementationType() const {
    const auto* pd = getPrimitiveDesc();
    return parse_impl_name(DnnlExtensionUtils::query_impl_info_str(pd));
}

}  // namespace ov::intel_cpu
