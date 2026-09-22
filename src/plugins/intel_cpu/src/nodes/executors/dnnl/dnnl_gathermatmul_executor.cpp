// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_gathermatmul_executor.hpp"

#include <oneapi/dnnl/dnnl_common_types.h>
#include <oneapi/dnnl/dnnl_types.h>

#include <algorithm>
#include <common/primitive_hashing.hpp>
#include <common/primitive_hashing_utils.hpp>
#include <common/utils.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nodes/common/blocked_desc_creator.h"
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include <cpu/x64/cpu_isa_traits.hpp>
#endif
#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/common/offset_helper.hpp"
#include "nodes/executors/dnnl/dnnl_utils.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/gathermatmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "thread_pool_imp.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

// ---- InnerProductKey ---------------------------------------------------------

struct InnerProductKey {
    dnnl::memory::desc src_md;
    dnnl::memory::desc weights_md;
    dnnl::memory::desc bias_md;
    VectorDims scale_shape;
    VectorDims zp_shape;
    // Native zero-point data type (as provided by the graph). Used as-is when
    // src-dynamic-quantization is enabled; otherwise zero points are converted to f32.
    dnnl::memory::data_type zp_dt = dnnl::memory::data_type::f32;
    // 0 disables src dynamic quantization (falls back to the general weights-decompression
    // path with f32 zero points); >0 is the src quantization group size along K.
    size_t src_dyn_quant_group_size = 0;

    [[nodiscard]] size_t hash() const {
        using namespace dnnl::impl;
        using namespace dnnl::impl::primitive_hashing;

        size_t seed = 0;
        seed = hash_combine(seed, get_md_hash(*src_md.get()));
        seed = hash_combine(seed, get_md_hash(*weights_md.get()));
        seed = hash_combine(seed, get_md_hash(*bias_md.get()));
        seed = get_vector_hash(seed, scale_shape);
        seed = get_vector_hash(seed, zp_shape);
        seed = hash_combine(seed, static_cast<size_t>(zp_dt));
        seed = hash_combine(seed, src_dyn_quant_group_size);
        return seed;
    }

    bool operator==(const InnerProductKey& rhs) const {
        return src_md == rhs.src_md && weights_md == rhs.weights_md && bias_md == rhs.bias_md &&
               scale_shape == rhs.scale_shape && zp_shape == rhs.zp_shape && zp_dt == rhs.zp_dt &&
               src_dyn_quant_group_size == rhs.src_dyn_quant_group_size;
    }
};

// ---- InnerProduct (oneDNN inner_product wrapper) ----------------------------

class GatherMatmulDnnlExecutor::InnerProduct {
public:
    InnerProduct() = delete;
    InnerProduct(const InnerProduct&) = delete;
    InnerProduct(InnerProduct&&) = delete;
    InnerProduct& operator=(const InnerProduct&) = delete;
    InnerProduct& operator=(InnerProduct&&) = delete;

    InnerProduct(const dnnl::engine& eng, const std::shared_ptr<ThreadPool>& threadPool, const InnerProductKey& key)
        : m_stream(make_stream(eng, threadPool)) {
        const auto& src_md = key.src_md;
        const auto& weights_md = key.weights_md;
        auto scale_shape = key.scale_shape;
        auto zp_shape = key.zp_shape;

        const auto weiDims = weights_md.get_dims();
        const auto K = weiDims[1];
        const auto N = weiDims[0];
        const auto M = src_md.get_dims()[0];

        if (!scale_shape.empty()) {
            if (all_of(1U, scale_shape.size(), scale_shape[0])) {
                scale_shape.push_back(1);
            }
            OPENVINO_ASSERT(scale_shape.size() == 2, "Unsupported scale shape ", vec2str(scale_shape));
            const auto K_groups = scale_shape.back();
            OPENVINO_ASSERT((K % K_groups) == 0, "Incompatible number of groups ", K_groups, " for K ", K);
            init_w_scales(scale_shape, weiDims);
            if (!zp_shape.empty()) {
                if (all_of(1U, zp_shape.size(), zp_shape[0])) {
                    zp_shape.push_back(1);
                }
                OPENVINO_ASSERT(zp_shape.size() == 2, "Unsupported zero points shape ", vec2str(zp_shape));
                init_w_zp(zp_shape,
                          weiDims,
                          key.src_dyn_quant_group_size > 0 ? key.zp_dt : dnnl::memory::data_type::f32);
            }
        }

        if (key.src_dyn_quant_group_size > 0) {
            m_attr.set_src_dyn_quant_params(key.src_dyn_quant_group_size);
        }

        m_input_md = src_md;
        m_output_md =
            dnnl::memory::desc(dnnl::memory::dims({M, N}), src_md.get_data_type(), dnnl::memory::format_tag::ab);

        const auto& bias_md = key.bias_md;

        auto ip_prim_desc = dnnl::inner_product_forward::primitive_desc(eng,
                                                                        dnnl::prop_kind::forward_inference,
                                                                        m_input_md,
                                                                        weights_md,
                                                                        bias_md,
                                                                        m_output_md,
                                                                        m_attr);

        m_impl_type = parse_impl_name(ip_prim_desc.impl_info_str());
        m_wei_md = ip_prim_desc.weights_desc();
        m_prim = dnnl::inner_product_forward(ip_prim_desc);

        if (std::getenv("U3DBG")) {
            printf("[U3DBG InnerProduct] K=%ld N=%ld M=%ld wei_dt=%d impl=%s\n", (long)K, (long)N, (long)M,
                   (int)weights_md.get_data_type(), ip_prim_desc.impl_info_str());
            fflush(stdout);
        }

        dnnl::memory inp_memory(m_input_md, eng, DNNL_MEMORY_NONE);
        dnnl::memory out_memory(m_output_md, eng, DNNL_MEMORY_NONE);
        dnnl::memory wei_memory(m_wei_md, eng, DNNL_MEMORY_NONE);
        dnnl::memory bias_memory;
        if (!bias_md.is_zero()) {
            bias_memory = dnnl::memory(bias_md, eng, DNNL_MEMORY_NONE);
        }
        dnnl::memory scale_memory;
        if (!scale_shape.empty()) {
            scale_memory = dnnl::memory(m_scale_md, eng, DNNL_MEMORY_NONE);
        }
        dnnl::memory zp_memory;
        if (!zp_shape.empty()) {
            zp_memory = dnnl::memory(m_zp_md, eng, DNNL_MEMORY_NONE);
        }
        m_args = make_args(inp_memory, out_memory, wei_memory, bias_memory, scale_memory, zp_memory);
    }

    void exec(void* src, void* dst, void* weight, void* bias = nullptr, void* scale = nullptr, void* zp = nullptr) {
        m_args[DNNL_ARG_SRC].set_data_handle(src);
        m_args[DNNL_ARG_DST].set_data_handle(dst);
        m_args[DNNL_ARG_WEIGHTS].set_data_handle(weight);
        if (bias) {
            m_args[DNNL_ARG_BIAS].set_data_handle(bias);
        }
        if (scale) {
            m_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS].set_data_handle(scale);
        }
        if (zp) {
            m_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS].set_data_handle(zp);
        }
        m_prim.execute(m_stream, m_args);
    }

    [[nodiscard]] dnnl::memory::desc get_weights_md() const {
        return m_wei_md;
    }
    [[nodiscard]] dnnl::memory::desc get_scale_md() const {
        return m_scale_md;
    }
    [[nodiscard]] dnnl::memory::desc get_zp_md() const {
        return m_zp_md;
    }
    [[nodiscard]] impl_desc_type get_impl_type() const {
        return m_impl_type;
    }

private:
    void init_w_scales(const VectorDims& scale_shape, const dnnl::memory::dims& /*weiDims*/) {
        constexpr auto data_type = dnnl::memory::data_type::f32;
        const auto scale_dims = DnnlExtensionUtils::convertToDnnlDims(scale_shape);
        m_attr.set_scales_dims(DNNL_ARG_WEIGHTS, scale_dims, data_type);
        m_scale_md = dnnl::memory::desc(scale_dims, data_type, dnnl::memory::format_tag::ba);
    }

    void init_w_zp(const VectorDims& zp_shape, const dnnl::memory::dims& /*weiDims*/, dnnl::memory::data_type data_type) {
        const auto zp_dims = DnnlExtensionUtils::convertToDnnlDims(zp_shape);
        m_attr.set_zero_points_dims(DNNL_ARG_WEIGHTS, zp_dims, data_type);
        m_zp_md = dnnl::memory::desc(zp_dims, data_type, dnnl::memory::format_tag::ba);
    }

    static std::unordered_map<int, dnnl::memory> make_args(dnnl::memory& src,
                                                           dnnl::memory& dst,
                                                           dnnl::memory& weight,
                                                           dnnl::memory& bias,
                                                           dnnl::memory& scale,
                                                           dnnl::memory& zp) {
        std::unordered_map<int, dnnl::memory> args;
        args.insert({DNNL_ARG_SRC, src});
        args.insert({DNNL_ARG_WEIGHTS, weight});
        args.insert({DNNL_ARG_DST, dst});
        if (bias) {
            args.insert({DNNL_ARG_BIAS, bias});
        }
        if (scale) {
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale});
        }
        if (zp) {
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp});
        }
        return args;
    }

    dnnl::stream m_stream;
    dnnl::primitive m_prim;
    dnnl::memory m_scratchpad_memory;
    dnnl::memory::desc m_input_md;
    dnnl::memory::desc m_output_md;
    dnnl::memory::desc m_wei_md;
    dnnl::memory::desc m_scale_md;
    dnnl::memory::desc m_zp_md;
    dnnl::primitive_attr m_attr;
    std::unordered_map<int, dnnl::memory> m_args;
    impl_desc_type m_impl_type = impl_desc_type::unknown;
};

namespace {

// ---- normalizeM helper -------------------------------------------------------

Dim normalizeM(Dim M) {
    if (M < 512) {
        M = rnd_up(M, 16);
    } else if (M < 1024) {
        M = rnd_up(M, 32);
    } else {
        M = rnd_up(M, 256);
    }
    return M;
}

// ---- GatherMatmulDnnlExecutor -----------------------------------------------

dnnl::memory::desc makeBiasMd(dnnl::memory::dim N, const MemoryPtr& biasMem) {
    if (!biasMem || biasMem->getDesc().empty()) {
        return {};
    }
    const auto bias_precision = biasMem->getDesc().getPrecision();
    return dnnl::memory::desc(dnnl::memory::dims({N}),
                              DnnlExtensionUtils::ElementTypeToDataType(bias_precision),
                              dnnl::memory::format_tag::a);
}
}  // namespace

bool GatherMatmulDnnlExecutor::supports([[maybe_unused]] const GatherMatmulConfig& config) {
#ifdef OPENVINO_ARCH_X86_64
    // Allow empty (dynamic) src descriptor — actual type is resolved at createPrimitive time
    if ((config.descs.count(ARG_SRC) != 0U) && !config.descs.at(ARG_SRC)->empty()) {
        const auto src_prc = config.descs.at(ARG_SRC)->getPrecision();
        if (!any_of(src_prc, ov::element::f32, ov::element::bf16)) {
            return false;
        }
    }
    // For compressed (int) weights, require AVX2
    if ((config.descs.count(ARG_WEI) != 0U) && !config.descs.at(ARG_WEI)->empty()) {
        const auto wei_prc = config.descs.at(ARG_WEI)->getPrecision();
        if (any_of(wei_prc,
                   ov::element::u8,
                   ov::element::i8,
                   ov::element::u4,
                   ov::element::i4,
                   ov::element::u3)) {
            if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
                return false;
            }
        }
    }
    return true;
#else
    return false;
#endif
}

GatherMatmulDnnlExecutor::GatherMatmulDnnlExecutor([[maybe_unused]] const GatherMatmulAttrs& attrs,
                                                   const MemoryArgs& memory,
                                                   const ExecutorContext::CPtr& context)
    : m_context(context) {
    const auto& weightsMemory = memory.at(ARG_WEI);
    const auto& srcMemory = memory.at(ARG_SRC);

    auto src_precision = srcMemory->getDesc().getPrecision();
    auto weights_precision = weightsMemory->getDesc().getPrecision();
#ifdef OPENVINO_ARCH_X86_64
    m_bf16AmxMode =
        (src_precision == ov::element::bf16 && dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx));
#endif

    const auto& weiDims = weightsMemory->getShape().getStaticDims();
    const dnnl::memory::dim N = weiDims[weiDims.size() - 2];
    const dnnl::memory::dim K = weiDims[weiDims.size() - 1];

    // Determine scale/zp shapes (per-group, removing the leading batch/gather dim)
    VectorDims scale_shape{};
    VectorDims zp_shape{};

    const auto& scalesMem = memory.at(ARG_SRC_3);
    if (scalesMem && !scalesMem->getDesc().empty()) {
        const auto& fullScalesShape = scalesMem->getShape().getStaticDims();
        if (1 == fullScalesShape.size()) {
            OPENVINO_ASSERT(fullScalesShape[0] == 1, "Expect broadcastable scales shape.");
            scale_shape.push_back(fullScalesShape[0]);
        } else {
            scale_shape.assign(fullScalesShape.begin() + 1, fullScalesShape.end());
        }
    }

    const auto& zpMem = memory.at(ARG_SRC_4);
    ov::element::Type zp_precision = ov::element::dynamic;
    if (zpMem && !zpMem->getDesc().empty()) {
        zp_precision = zpMem->getDesc().getPrecision();
        const auto& fullZpShape = zpMem->getShape().getStaticDims();
        if (1 == fullZpShape.size()) {
            OPENVINO_ASSERT(fullZpShape[0] == 1, "Expect broadcastable zero points shape.");
            zp_shape.push_back(fullZpShape[0]);
        } else {
            zp_shape.assign(fullZpShape.begin() + 1, fullZpShape.end());
        }
    }

    if (std::getenv("U3DBG")) {
        m_nativeWeightsMemoryDbg = weightsMemory;
        m_nativeScalesMemoryDbg = scalesMem;
        m_nativeZpMemoryDbg = zpMem;
        m_weightsPrecisionDbg = weights_precision;
        m_zpPrecisionDbg = zp_precision;
    }

    // U3OVERRIDE: overwrite weight/scale/zp with a deterministic, dtype-agnostic value pattern
    // (values in [0,7], representable identically in u3 and u4 native packing) so that u3 and
    // u4 runs can be directly diffed step by step with truly identical logical inputs.
    if (std::getenv("U3OVERRIDE")) {
        auto set_native = [](const MemoryPtr& mem, ov::element::Type prc, size_t E, size_t OC, size_t G,
                              const std::function<uint8_t(size_t, size_t, size_t)>& valueFn) {
            auto* raw = static_cast<uint8_t*>(mem->getData());
            auto set_bits = [&](size_t idx, uint8_t val, int nbits) {
                size_t bit = idx * nbits;
                size_t byte = bit / 8;
                size_t shift = bit % 8;
                uint16_t cur = raw[byte];
                if (shift + nbits > 8) {
                    cur |= static_cast<uint16_t>(raw[byte + 1]) << 8;
                }
                uint16_t mask = static_cast<uint16_t>((1U << nbits) - 1U) << shift;
                cur = (cur & ~mask) | ((static_cast<uint16_t>(val) << shift) & mask);
                raw[byte] = cur & 0xFF;
                if (shift + nbits > 8) {
                    raw[byte + 1] = (cur >> 8) & 0xFF;
                }
            };
            for (size_t e = 0; e < E; e++) {
                for (size_t oc = 0; oc < OC; oc++) {
                    for (size_t g = 0; g < G; g++) {
                        uint8_t v = valueFn(e, oc, g);
                        size_t idx = e * OC * G + oc * G + g;
                        if (prc == ov::element::u3) {
                            set_bits(idx, v & 0x7U, 3);
                        } else if (prc == ov::element::u4) {
                            set_bits(idx, v & 0xFU, 4);
                        }
                    }
                }
            }
        };
        if (std::string(std::getenv("U3OVERRIDE")) == "2") {
            // Fully random (fixed-seed, reproducible) weight/scale/zp/src -- no closed-form
            // expected value here; rely on the decode_native-based hand-ref check in execute()
            // (which reads whatever is actually in the native buffers) to validate.
            // FIX_WEIGHT/FIX_ZP/FIX_SCALE env vars pin one dimension to a simple constant while
            // the rest stay random, to bisect which dimension's randomness is actually required
            // to reproduce the divergence (paired with FIX_SRC handled in execute()).
            std::srand(12345);
            const bool fixWeight = std::getenv("FIX_WEIGHT") != nullptr;
            const bool fixZp = std::getenv("FIX_ZP") != nullptr;
            const bool fixScale = std::getenv("FIX_SCALE") != nullptr;
            set_native(weightsMemory, weights_precision, weiDims[0], N, K,
                       [fixWeight](size_t, size_t, size_t ic) {
                           return fixWeight ? static_cast<uint8_t>(ic % 8) : static_cast<uint8_t>(std::rand() % 8);
                       });
            if (scalesMem && !scalesMem->getDesc().empty()) {
                const auto& scDims = scalesMem->getShape().getStaticDims();
                auto* scaleF = scalesMem->getDataAs<float>();
                for (size_t i = 0; i < scDims[0] * scDims[1] * scDims[2]; i++) {
                    scaleF[i] = fixScale ? 1.0f : 0.001f + 0.05f * (static_cast<float>(std::rand()) / RAND_MAX);
                }
            }
            if (zpMem && !zpMem->getDesc().empty()) {
                const auto& zDims = zpMem->getShape().getStaticDims();
                set_native(zpMem, zp_precision, zDims[0], zDims[1], zDims[2],
                           [fixZp](size_t, size_t, size_t g) {
                               return fixZp ? static_cast<uint8_t>(0) : static_cast<uint8_t>(std::rand() % 8);
                           });
            }
            printf("[U3DBG override] RANDOM MODE wei_prc=%s zp_prc=%s N=%ld K=%ld fixWeight=%d fixZp=%d fixScale=%d\n",
                   weights_precision.get_type_name().c_str(), zp_precision.get_type_name().c_str(), (long)N, (long)K,
                   fixWeight, fixZp, fixScale);
            fflush(stdout);
        } else if (std::string(std::getenv("U3OVERRIDE")) == "3") {
            // Isolate OC-dependence: weight[oc,g] = (oc+g)%8 (constant WITHIN a group, but shifts
            // with oc -- unlike all earlier modes where weight/scale/zp never depended on oc).
            // scale_g=1+0.1*g and zp_g=g stay OC-INVARIANT, matching the earlier passing mode, so
            // any divergence here isolates an OC-indexing bug specifically in the WEIGHT buffer.
            long numGroups3 = scalesMem ? static_cast<long>(scalesMem->getShape().getStaticDims()[2]) : 1;
            long groupSize3 = static_cast<long>(K) / numGroups3;
            set_native(weightsMemory, weights_precision, weiDims[0], N, K,
                       [groupSize3](size_t /*e*/, size_t oc, size_t ic) {
                           long g = static_cast<long>(ic) / groupSize3;
                           return static_cast<uint8_t>((oc + static_cast<size_t>(g)) % 8);
                       });
            if (scalesMem && !scalesMem->getDesc().empty()) {
                const auto& scDims = scalesMem->getShape().getStaticDims();
                auto* scaleF = scalesMem->getDataAs<float>();
                for (size_t e = 0; e < scDims[0]; e++) {
                    for (size_t oc = 0; oc < scDims[1]; oc++) {
                        for (size_t g = 0; g < scDims[2]; g++) {
                            scaleF[e * scDims[1] * scDims[2] + oc * scDims[2] + g] =
                                1.0f + 0.1f * static_cast<float>(g);
                        }
                    }
                }
            }
            if (zpMem && !zpMem->getDesc().empty()) {
                const auto& zDims = zpMem->getShape().getStaticDims();
                set_native(zpMem, zp_precision, zDims[0], zDims[1], zDims[2],
                           [](size_t /*e*/, size_t /*oc*/, size_t g) { return static_cast<uint8_t>(g); });
            }
            for (long oc = 0; oc < 3 && oc < N; oc++) {
                double expected3 = 0.0;
                for (long g = 0; g < numGroups3; g++) {
                    double scaleV = 1.0 + 0.1 * static_cast<double>(g);
                    double wv = static_cast<double>((oc + g) % 8);
                    expected3 += static_cast<double>(groupSize3) * (wv - static_cast<double>(g)) * scaleV;
                }
                printf("[U3DBG override] MODE3 wei_prc=%s N=%ld K=%ld oc=%ld EXPECTED=%f\n",
                       weights_precision.get_type_name().c_str(), (long)N, (long)K, oc, expected3);
            }
            fflush(stdout);
        } else if (std::string(std::getenv("U3OVERRIDE")) == "4") {
            // Weight now varies BOTH per-ic WITHIN a group AND per-oc (unlike mode 3, which was
            // constant within each group). scale_g=1+0.1*g and zp_g=g stay OC-INVARIANT, same as
            // the passing baseline, so any divergence here isolates the WEIGHT buffer's combined
            // (oc, in-group ic) indexing as the bug, distinct from mode 3's oc-only variation.
            long numGroups4 = scalesMem ? static_cast<long>(scalesMem->getShape().getStaticDims()[2]) : 1;
            long groupSize4 = static_cast<long>(K) / numGroups4;
            auto weightFn4 = [](size_t oc, size_t ic) { return static_cast<uint8_t>((oc * 7 + ic * 3) % 8); };
            set_native(weightsMemory, weights_precision, weiDims[0], N, K,
                       [weightFn4](size_t /*e*/, size_t oc, size_t ic) { return weightFn4(oc, ic); });
            if (scalesMem && !scalesMem->getDesc().empty()) {
                const auto& scDims = scalesMem->getShape().getStaticDims();
                auto* scaleF = scalesMem->getDataAs<float>();
                for (size_t e = 0; e < scDims[0]; e++) {
                    for (size_t oc = 0; oc < scDims[1]; oc++) {
                        for (size_t g = 0; g < scDims[2]; g++) {
                            scaleF[e * scDims[1] * scDims[2] + oc * scDims[2] + g] =
                                1.0f + 0.1f * static_cast<float>(g);
                        }
                    }
                }
            }
            if (zpMem && !zpMem->getDesc().empty()) {
                const auto& zDims = zpMem->getShape().getStaticDims();
                set_native(zpMem, zp_precision, zDims[0], zDims[1], zDims[2],
                           [](size_t /*e*/, size_t /*oc*/, size_t g) { return static_cast<uint8_t>(g); });
            }
            for (long oc = 0; oc < 3 && oc < N; oc++) {
                double expected4 = 0.0;
                for (long g = 0; g < numGroups4; g++) {
                    double scaleV = 1.0 + 0.1 * static_cast<double>(g);
                    double weightSum = 0.0;
                    for (long ic = g * groupSize4; ic < (g + 1) * groupSize4; ic++) {
                        weightSum += static_cast<double>(weightFn4(static_cast<size_t>(oc), static_cast<size_t>(ic)));
                    }
                    expected4 += (weightSum - static_cast<double>(groupSize4) * static_cast<double>(g)) * scaleV;
                }
                printf("[U3DBG override] MODE4 wei_prc=%s N=%ld K=%ld oc=%ld EXPECTED=%f\n",
                       weights_precision.get_type_name().c_str(), (long)N, (long)K, oc, expected4);
            }
            fflush(stdout);
        } else if (std::string(std::getenv("U3OVERRIDE")) == "5") {
            // Isolate WEIGHT randomness alone: weight is genuinely random (fixed seed) per
            // (oc,ic), while scale=1.0 and zp=0 are held FIXED CONSTANTS (not even per-group
            // variation). If this alone reproduces the mode-2 divergence, the bug is in how the
            // weight buffer's actual (non-repeating) bit patterns get decoded/consumed -- not an
            // interaction with scale/zp variation.
            std::srand(12345);
            std::vector<uint8_t> weiVals4(static_cast<size_t>(N) * static_cast<size_t>(K));
            for (auto& v : weiVals4) {
                v = static_cast<uint8_t>(std::rand() % 8);
            }
            set_native(weightsMemory, weights_precision, weiDims[0], N, K,
                       [&weiVals4, K](size_t /*e*/, size_t oc, size_t ic) {
                           return weiVals4[oc * static_cast<size_t>(K) + ic];
                       });
            if (scalesMem && !scalesMem->getDesc().empty()) {
                const auto& scDims = scalesMem->getShape().getStaticDims();
                auto* scaleF = scalesMem->getDataAs<float>();
                for (size_t i = 0; i < scDims[0] * scDims[1] * scDims[2]; i++) {
                    scaleF[i] = 1.0f;
                }
            }
            if (zpMem && !zpMem->getDesc().empty()) {
                const auto& zDims = zpMem->getShape().getStaticDims();
                set_native(zpMem, zp_precision, zDims[0], zDims[1], zDims[2],
                           [](size_t, size_t, size_t) { return static_cast<uint8_t>(0); });
            }
            for (long oc = 0; oc < 3 && oc < N; oc++) {
                double expected5 = 0.0;
                for (long ic = 0; ic < K; ic++) {
                    expected5 += static_cast<double>(weiVals4[static_cast<size_t>(oc) * static_cast<size_t>(K) +
                                                               static_cast<size_t>(ic)]);
                }
                printf("[U3DBG override] MODE5 wei_prc=%s N=%ld K=%ld oc=%ld EXPECTED=%f\n",
                       weights_precision.get_type_name().c_str(), (long)N, (long)K, oc, expected5);
            }
            fflush(stdout);
        } else if (std::string(std::getenv("U3OVERRIDE")) == "6") {
            // Isolate ZP randomness alone: weight=ic%8 (simple), scale=1.0 fixed, zp is genuinely
            // random per (oc,group) (fixed seed). If this alone reproduces mode-2's divergence,
            // the bug is specifically in how the ZP buffer's per-oc/per-group values are consumed.
            std::srand(12345);
            const auto& scDims6 = scalesMem->getShape().getStaticDims();  // [E, OC, groups]
            std::vector<uint8_t> zpVals6(scDims6[0] * scDims6[1] * scDims6[2]);
            for (auto& v : zpVals6) {
                v = static_cast<uint8_t>(std::rand() % 8);
            }
            set_native(weightsMemory, weights_precision, weiDims[0], N, K,
                       [](size_t /*e*/, size_t /*oc*/, size_t ic) { return static_cast<uint8_t>(ic % 8); });
            auto* scaleF6 = scalesMem->getDataAs<float>();
            for (size_t i = 0; i < scDims6[0] * scDims6[1] * scDims6[2]; i++) {
                scaleF6[i] = 1.0f;
            }
            if (zpMem && !zpMem->getDesc().empty()) {
                const auto& zDims = zpMem->getShape().getStaticDims();
                set_native(zpMem, zp_precision, zDims[0], zDims[1], zDims[2],
                           [&zpVals6, &scDims6](size_t e, size_t oc, size_t g) {
                               return zpVals6[e * scDims6[1] * scDims6[2] + oc * scDims6[2] + g];
                           });
            }
            long numGroups6 = static_cast<long>(scDims6[2]);
            long groupSize6 = static_cast<long>(K) / numGroups6;
            double weightSumPerGroup6 = (static_cast<double>(groupSize6) / 8.0) * 28.0;
            for (long oc = 0; oc < 3 && oc < N; oc++) {
                double expected6 = 0.0;
                for (long g = 0; g < numGroups6; g++) {
                    double zpV = static_cast<double>(
                        zpVals6[2 * scDims6[1] * scDims6[2] + static_cast<size_t>(oc) * scDims6[2] +
                                static_cast<size_t>(g)]);
                    expected6 += weightSumPerGroup6 - static_cast<double>(groupSize6) * zpV;
                }
                printf("[U3DBG override] MODE6 wei_prc=%s N=%ld K=%ld oc=%ld EXPECTED(ge=2)=%f\n",
                       weights_precision.get_type_name().c_str(), (long)N, (long)K, oc, expected6);
            }
            fflush(stdout);
        } else if (std::string(std::getenv("U3OVERRIDE")) == "7") {
            // Isolate SCALE randomness alone: weight=ic%8 (simple), zp=0 fixed, scale is genuinely
            // random per (oc,group) in the SAME small range used by mode 2 ([0.001,0.051]).
            std::srand(12345);
            const auto& scDims7 = scalesMem->getShape().getStaticDims();
            std::vector<float> scaleVals7(scDims7[0] * scDims7[1] * scDims7[2]);
            for (auto& v : scaleVals7) {
                v = 0.001f + 0.05f * (static_cast<float>(std::rand()) / RAND_MAX);
            }
            set_native(weightsMemory, weights_precision, weiDims[0], N, K,
                       [](size_t /*e*/, size_t /*oc*/, size_t ic) { return static_cast<uint8_t>(ic % 8); });
            auto* scaleF7 = scalesMem->getDataAs<float>();
            for (size_t i = 0; i < scaleVals7.size(); i++) {
                scaleF7[i] = scaleVals7[i];
            }
            if (zpMem && !zpMem->getDesc().empty()) {
                const auto& zDims = zpMem->getShape().getStaticDims();
                set_native(zpMem, zp_precision, zDims[0], zDims[1], zDims[2],
                           [](size_t, size_t, size_t) { return static_cast<uint8_t>(0); });
            }
            long numGroups7 = static_cast<long>(scDims7[2]);
            long groupSize7 = static_cast<long>(K) / numGroups7;
            double weightSumPerGroup7 = (static_cast<double>(groupSize7) / 8.0) * 28.0;
            for (long oc = 0; oc < 3 && oc < N; oc++) {
                double expected7 = 0.0;
                for (long g = 0; g < numGroups7; g++) {
                    double scaleV = static_cast<double>(
                        scaleVals7[2 * scDims7[1] * scDims7[2] + static_cast<size_t>(oc) * scDims7[2] +
                                   static_cast<size_t>(g)]);
                    expected7 += weightSumPerGroup7 * scaleV;
                }
                printf("[U3DBG override] MODE7 wei_prc=%s N=%ld K=%ld oc=%ld EXPECTED(ge=2)=%f\n",
                       weights_precision.get_type_name().c_str(), (long)N, (long)K, oc, expected7);
            }
            fflush(stdout);
        } else {
        // Trivially hand-computable values: weight cycles 0..7 WITHIN each group (per-element
        // variation, not just per-group), scale=1 everywhere, zp=g (group index) per group, src=1
        // everywhere. Each full 0..7 cycle sums to 28; groupSize=32 is 4 full cycles, so each
        // group's weight sum is 4*28=112 regardless of g. Expected per-oc output is then
        // sum_g(112 - groupSize*g), independent of oc/expert -- easy to verify by eye.
        set_native(weightsMemory, weights_precision, weiDims[0], N, K,
                   [](size_t /*e*/, size_t /*oc*/, size_t ic) { return static_cast<uint8_t>(ic % 8); });
        // Scale now varies PER GROUP too: scale_g = 1 + 0.1*g (still trivially hand-computable).
        if (scalesMem && !scalesMem->getDesc().empty()) {
            const auto& scDims = scalesMem->getShape().getStaticDims();  // [E, OC, groups]
            auto* scaleF = scalesMem->getDataAs<float>();
            for (size_t e = 0; e < scDims[0]; e++) {
                for (size_t oc = 0; oc < scDims[1]; oc++) {
                    for (size_t g = 0; g < scDims[2]; g++) {
                        scaleF[e * scDims[1] * scDims[2] + oc * scDims[2] + g] = 1.0f + 0.1f * static_cast<float>(g);
                    }
                }
            }
        }
        if (zpMem && !zpMem->getDesc().empty()) {
            const auto& zDims = zpMem->getShape().getStaticDims();
            set_native(zpMem, zp_precision, zDims[0], zDims[1], zDims[2],
                       [](size_t /*e*/, size_t /*oc*/, size_t g) { return static_cast<uint8_t>(g); });
        }
        // Hand-computable expected value: weight cycles 0..7 within each group of size groupSize
        // (groupSize is a multiple of 8 for all tested shapes, so each group's raw weight sum is
        // exactly (groupSize/8)*28), scale_g=1+0.1*g, zp_g=g, src=1 everywhere. Each group
        // contributes (weightSum_g - groupSize*g) * scale_g to the output, for every oc/expert.
        long numGroups = scalesMem ? static_cast<long>(scalesMem->getShape().getStaticDims()[2]) : 1;
        long groupSize = static_cast<long>(K) / numGroups;
        double weightSumPerGroup = (static_cast<double>(groupSize) / 8.0) * 28.0;
        double expected = 0.0;
        for (long g = 0; g < numGroups; g++) {
            double scaleV = 1.0 + 0.1 * static_cast<double>(g);
            expected += (weightSumPerGroup - static_cast<double>(groupSize) * static_cast<double>(g)) * scaleV;
        }
        printf("[U3DBG override] wei_prc=%s zp_prc=%s N=%ld K=%ld numGroups=%ld groupSize=%ld EXPECTED=%f\n",
               weights_precision.get_type_name().c_str(), zp_precision.get_type_name().c_str(), (long)N, (long)K,
               numGroups, groupSize, expected);
        fflush(stdout);
        }
    }

    dnnl::memory::desc src_md({1, K},
                              DnnlExtensionUtils::ElementTypeToDataType(src_precision),
                              dnnl::memory::format_tag::ab);
    dnnl::memory::desc weights_md({N, K},
                                  DnnlExtensionUtils::ElementTypeToDataType(weights_precision),
                                  dnnl::memory::format_tag::any);

    // U3 has no working non-dyn-quant (general weights-decompression) kernel, so route it
    // through src dynamic quantization (VNNI) instead, mirroring FullyConnected's approach.
    // Gated to u3 specifically to avoid touching the already-working u4/u8/i4 paths.
    size_t src_dyn_quant_group_size = 0;
    auto zp_dt = dnnl::memory::data_type::f32;
    if (weights_precision == ov::element::u3 &&
        (ov::with_cpu_x86_avx2_vnni() || ov::with_cpu_x86_avx512_core_vnni()) &&
        any_of(zp_precision, ov::element::u8, ov::element::u2, ov::element::u3, ov::element::u4, ov::element::dynamic)) {
        constexpr size_t default_group_size = 32;
        src_dyn_quant_group_size = default_group_size;
        if (zp_precision != ov::element::dynamic) {
            // Force u8 rather than passing the native zero-point precision through: oneDNN's
            // zero-points attr wants the (possibly batched) zp memory in a transposed layout
            // (format_tag::ba), and no reorder implementation exists anywhere in this codebase
            // for a same-native-sub-byte-precision, arbitrary-permutation transform (verified:
            // simple_reorder.hpp / cpu_reorder_regular_{u2,u3,u4,s4}.cpp). Converting to u8
            // here instead makes the reorder a precision-converting one (<native> -> u8), which
            // is well supported via cpu_convert's fallback in node::Reorder::reorderData — the
            // same path FullyConnected's own dyn-quant zero-point handling always uses
            // (dnnl_fullyconnected_primitive.cpp: dstPrc = useDynamicQuantization ? u8 : f32).
            // Safe: u3's zero-point value range [0,7] fits u8 exactly.
            zp_dt = dnnl::memory::data_type::u8;
        }
    }

    InnerProductKey key{src_md,
                       weights_md,
                       makeBiasMd(N, memory.at(ARG_BIAS)),
                       scale_shape,
                       zp_shape,
                       zp_dt,
                       src_dyn_quant_group_size};

    const auto& eng = context->getEngine();
    const auto threadPool = context->getThreadPool();
    auto cache = context->getRuntimeCache();
    std::tie(m_gemvImpl, std::ignore) = cache->getOrCreate(key, [&eng, &threadPool](const InnerProductKey& k) {
        return std::make_shared<InnerProduct>(eng, threadPool, k);
    });

    // Repack weights: convert from [G, K, N] to [G, (packed_N, K)] format expected by oneDNN
    auto gemvWeightsDesc =
        MemoryDescUtils::convertToBlockedMemoryDesc(DnnlExtensionUtils::makeDescriptor(m_gemvImpl->get_weights_md()));

    auto addBatchDim = [](const BlockedMemoryDescPtr& desc, size_t batchDim) -> DnnlMemoryDescPtr {
        const auto& weightsDims = desc->getShape().getStaticDims();
        const auto& weightsBlockDims = desc->getBlockDims();
        const auto& weightsOrder = desc->getOrder();
        VectorDims newDims = {batchDim};
        newDims.insert(newDims.end(), weightsDims.begin(), weightsDims.end());
        VectorDims newBlockDims = {batchDim};
        newBlockDims.insert(newBlockDims.end(), weightsBlockDims.begin(), weightsBlockDims.end());
        VectorDims newOrder(weightsOrder.size() + 1);
        newOrder[0] = 0;
        for (size_t i = 0; i < weightsOrder.size(); i++) {
            newOrder[i + 1] = weightsOrder[i] + 1;
        }
        auto targetDesc =
            std::make_shared<CpuBlockedMemoryDesc>(desc->getPrecision(), Shape(newDims), newBlockDims, newOrder);
        return MemoryDescUtils::convertToDnnlMemoryDesc(targetDesc);
    };

    auto targetWeightsDesc = addBatchDim(gemvWeightsDesc, weiDims[0]);
    auto srcWeightsDesc = MemoryDescUtils::convertToDnnlMemoryDesc(weightsMemory->getDescPtr());

    m_weightsMemory = utils::prepareWeightsMemory(srcWeightsDesc,
                                                  targetWeightsDesc,
                                                  weightsMemory,
                                                  eng,
                                                  cache,
                                                  context->getWeightsCache(),
                                                  context->getPrivateWeightCache(),
                                                  threadPool);

    if (scalesMem && !scale_shape.empty()) {
        auto expectedScaleMemDesc =
            MemoryDescUtils::convertToDnnlMemoryDesc(DnnlExtensionUtils::makeDescriptor(m_gemvImpl->get_scale_md()));
        const auto& scDims = scalesMem->getShape().getStaticDims();
        expectedScaleMemDesc =
            addBatchDim(MemoryDescUtils::convertToBlockedMemoryDesc(expectedScaleMemDesc), scDims[0]);
        if (expectedScaleMemDesc->isCompatible(scalesMem->getDesc())) {
            m_scalesMemory = std::const_pointer_cast<IMemory>(scalesMem);
        } else {
            m_scalesMemory = std::make_shared<Memory>(eng, expectedScaleMemDesc);
            m_scalesMemory->load(*scalesMem, false, false);
        }
    }

    if (zpMem && !zp_shape.empty()) {
        auto expectedZpMemDesc =
            MemoryDescUtils::convertToDnnlMemoryDesc(DnnlExtensionUtils::makeDescriptor(m_gemvImpl->get_zp_md()));
        const auto& zpDims = zpMem->getShape().getStaticDims();
        expectedZpMemDesc = addBatchDim(MemoryDescUtils::convertToBlockedMemoryDesc(expectedZpMemDesc), zpDims[0]);
        if (expectedZpMemDesc->isCompatible(zpMem->getDesc())) {
            m_zpMemory = std::const_pointer_cast<IMemory>(zpMem);
        } else {
            m_zpMemory = std::make_shared<Memory>(eng, expectedZpMemDesc);
            m_zpMemory->load(*zpMem, false, false);
        }
    }

    m_implType = m_gemvImpl->get_impl_type();
}

bool GatherMatmulDnnlExecutor::update(const MemoryArgs& memory) {
    if (!m_bf16AmxMode) {
        return true;
    }

    const auto& srcMem = memory.at(ARG_SRC);
    const auto& srcShape = srcMem->getStaticDims();
    // srcShape is [B, M, K]
    if (Dim{1} == srcShape[1]) {
        // If M is 1, we can skip the temporary buffer and execute GEMV in-place on the src buffer
        return true;
    }
    const Dim M = normalizeM(srcShape[1]);
    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    const auto srcPrc = srcMem->getDesc().getPrecision();

    const auto& dstMem = memory.at(ARG_DST);
    const auto& dstShape = dstMem->getStaticDims();

    m_tmpInputDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcPrc, Shape({M, srcShape[2]}));
    m_tmpOutputDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcPrc, Shape({M, dstShape[2]}));

    const size_t srcSize = rnd_up(m_tmpInputDesc->getCurrentMemSize(), 64);
    const size_t totalSize = srcSize + m_tmpOutputDesc->getCurrentMemSize();
    auto scratchPadDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(ov::element::u8, Shape({totalSize}));
    m_tmpInpBuffer = m_context->getScratchPad()->createScratchPadMem(scratchPadDesc);

    OPENVINO_ASSERT(m_gemvImpl, "GEMV implementation is not created");

    dnnl::memory::desc src_md({static_cast<dnnl::memory::dim>(M), static_cast<dnnl::memory::dim>(srcShape[2])},
                              DnnlExtensionUtils::ElementTypeToDataType(srcPrc),
                              dnnl::memory::format_tag::ab);
    auto weights_md = m_gemvImpl->get_weights_md();

    VectorDims scale_shape{};
    VectorDims zp_shape{};
    if (m_scalesMemory) {
        const auto& fullScaleDims = m_scalesMemory->getStaticDims();
        if (1 == fullScaleDims.size()) {
            scale_shape.push_back(fullScaleDims[0]);
        } else {
            scale_shape.assign(fullScaleDims.begin() + 1, fullScaleDims.end());
        }
    }
    ov::element::Type zp_precision = ov::element::dynamic;
    if (m_zpMemory) {
        zp_precision = m_zpMemory->getDesc().getPrecision();
        const auto& fullZpDims = m_zpMemory->getStaticDims();
        if (1 == fullZpDims.size()) {
            zp_shape.push_back(fullZpDims[0]);
        } else {
            zp_shape.assign(fullZpDims.begin() + 1, fullZpDims.end());
        }
    }

    // See matching comment at the GEMV InnerProductKey construction above (forcing u8 to avoid
    // the missing same-native-sub-byte-precision, arbitrary-permutation reorder capability).
    size_t src_dyn_quant_group_size = 0;
    auto zp_dt = dnnl::memory::data_type::f32;
    if (weights_md.get_data_type() == dnnl::memory::data_type::u3 &&
        (ov::with_cpu_x86_avx2_vnni() || ov::with_cpu_x86_avx512_core_vnni()) &&
        any_of(zp_precision, ov::element::u8, ov::element::u2, ov::element::u3, ov::element::u4, ov::element::dynamic)) {
        constexpr size_t default_group_size = 32;
        src_dyn_quant_group_size = default_group_size;
        if (zp_precision != ov::element::dynamic) {
            zp_dt = dnnl::memory::data_type::u8;
        }
    }

    InnerProductKey key{src_md,
                        weights_md,
                        makeBiasMd(static_cast<dnnl::memory::dim>(weights_md.get_dims()[0]), memory.at(ARG_BIAS)),
                        scale_shape,
                        zp_shape,
                        zp_dt,
                        src_dyn_quant_group_size};
    const auto& eng = m_context->getEngine();
    const auto threadPool = m_context->getThreadPool();
    auto cache = m_context->getRuntimeCache();
    std::tie(m_gemmImpl, std::ignore) = cache->getOrCreate(key, [&eng, &threadPool](const InnerProductKey& k) {
        return std::make_shared<InnerProduct>(eng, threadPool, k);
    });
    return true;
}

void GatherMatmulDnnlExecutor::execute(const MemoryArgs& memory) {
    const auto& cpu_parallel = m_context->getCpuParallel();
    const auto& srcMem = memory.at(ARG_SRC);
    const auto& biasMem = memory.at(ARG_BIAS);
    const auto& indexMem = memory.at(ARG_SRC_1);
    const auto& dstMem = memory.at(ARG_DST);

    const auto& indexShape = indexMem->getStaticDims();
    size_t M = indexShape[0];
    size_t indices_size = indexShape[1];

    auto src_offset = OffsetHelper::createOffsetHelper(srcMem);
    auto dst_offset = OffsetHelper::createOffsetHelper(dstMem);
    auto wei_offset = OffsetHelper::createOffsetHelper(m_weightsMemory);
    auto bias_offset = OffsetHelper::createOffsetHelper(biasMem);
    auto scale_offset = OffsetHelper::createOffsetHelper(m_scalesMemory);
    auto zp_offset = OffsetHelper::createOffsetHelper(m_zpMemory);
    auto index_offset = OffsetHelper::createOffsetHelper(indexMem);

    const size_t gather_axis_size = m_weightsMemory->getStaticDims()[0];

    // U3OVERRIDE: overwrite src with a deterministic pattern (independent of dtype) so u3/u4 runs
    // are directly comparable end to end (paired with the constructor's weight/scale/zp override).
    if (std::getenv("U3OVERRIDE")) {
        auto* srcF = srcMem->getDataAs<float>();
        const size_t total = srcMem->getSize() / sizeof(float);
        const std::string overrideMode = std::getenv("U3OVERRIDE");
        const bool fixSrc = std::getenv("FIX_SRC") != nullptr;
        const bool randomMode = (overrideMode == "2" && !fixSrc) || overrideMode == "8";
        for (size_t idx = 0; idx < total; idx++) {
            srcF[idx] = randomMode ? (-1.0f + 2.0f * (static_cast<float>(std::rand()) / RAND_MAX)) : 1.0f;
        }
    }

    if (M > 1) {
        std::vector<std::pair<int32_t, int32_t>> gather_idx_map(gather_axis_size * M);
        std::vector<int32_t> elements_per_gather_indx(gather_axis_size, 0);
        for (size_t m = 0; m < M; m++) {
            const auto* gather_ids = static_cast<const int32_t*>(index_offset(m));
            for (size_t i = 0; i < indices_size; i++) {
                int32_t gather_axis_index = gather_ids[i];
                OPENVINO_ASSERT(gather_axis_index >= 0 && static_cast<size_t>(gather_axis_index) < gather_axis_size,
                                "Invalid gather_id ",
                                gather_axis_index,
                                " for m ",
                                m);
                auto& index = elements_per_gather_indx[gather_axis_index];
                gather_idx_map[gather_axis_index * M + index] = {m, i};
                index++;
            }
        }

        if (m_bf16AmxMode) {
            OPENVINO_ASSERT(m_tmpInpBuffer, "Temporary input/output memory is not created");
            OPENVINO_ASSERT(m_tmpInputDesc, "Temporary input memory desc is not created");
            OPENVINO_ASSERT(m_tmpOutputDesc, "Temporary output memory desc is not created");

            const auto element_size = m_tmpInputDesc->getPrecision().size();
            const auto K_size = m_tmpInputDesc->getShape().getStaticDims()[1];
            const auto M_size = m_tmpInputDesc->getShape().getStaticDims()[0];
            const auto N_size = dstMem->getStaticDims()[2];

            auto* input_ptr = m_tmpInpBuffer->getDataAs<uint8_t>();
            auto* output_ptr = input_ptr + rnd_up(m_tmpInputDesc->getCurrentMemSize(), 64);

            Memory tmpInput(m_context->getEngine(), m_tmpInputDesc, m_tmpInpBuffer->getMemoryBlock());
            Memory tmpOutput(m_context->getEngine(), m_tmpOutputDesc, output_ptr);

            auto tmp_input_offset = OffsetHelper::createOffsetHelper(tmpInput);
            auto tmp_dst_offset = OffsetHelper::createOffsetHelper(tmpOutput);

            OPENVINO_ASSERT(m_gemmImpl, "GEMM implementation is not created");
            for (size_t gather_axis_index = 0; gather_axis_index < gather_axis_size; gather_axis_index++) {
                const size_t num_valid_rows = elements_per_gather_indx[gather_axis_index];
                if (0 == num_valid_rows) {
                    continue;
                }

                cpu_parallel->parallel_for(M_size, [&](size_t m) {
                    auto* dst_row = tmp_input_offset(m);
                    if (m < num_valid_rows) {
                        const auto row_id = gather_idx_map[gather_axis_index * M + m].first;
                        const auto batch_index = gather_idx_map[gather_axis_index * M + m].second;
                        const auto* src_data = src_offset(batch_index, row_id);
                        std::memcpy(dst_row, src_data, K_size * element_size);
                    } else {
                        std::memset(dst_row, 0, K_size * element_size);
                    }
                });

                auto* src = tmp_input_offset.get_base();
                auto* dst = tmp_dst_offset.get_base();
                auto* wei = wei_offset(gather_axis_index);
                auto* bias = bias_offset(gather_axis_index);
                auto* scale = scale_offset(gather_axis_index);
                auto* zp = zp_offset(gather_axis_index);
                m_gemmImpl->exec(src, dst, wei, bias, scale, zp);

                cpu_parallel->parallel_for(num_valid_rows, [&](size_t m) {
                    const auto* src_row = tmp_dst_offset(m);
                    const auto row_id = gather_idx_map[gather_axis_index * M + m].first;
                    const auto batch_index = gather_idx_map[gather_axis_index * M + m].second;
                    auto* dst_row = dst_offset(batch_index, row_id);
                    std::memcpy(dst_row, src_row, N_size * element_size);
                });
            }
        } else {
            OPENVINO_ASSERT(m_gemvImpl, "GEMV implementation is not created");
            for (size_t gather_axis_index = 0; gather_axis_index < gather_axis_size; gather_axis_index++) {
                if (0 == elements_per_gather_indx[gather_axis_index]) {
                    continue;
                }
                auto* wei = wei_offset(gather_axis_index);
                auto* bias = bias_offset(gather_axis_index);
                auto* scale = scale_offset(gather_axis_index);
                auto* zp = zp_offset(gather_axis_index);
                for (int32_t m = 0; m < elements_per_gather_indx[gather_axis_index]; ++m) {
                    const auto row_id = gather_idx_map[gather_axis_index * M + m].first;
                    const auto batch_index = gather_idx_map[gather_axis_index * M + m].second;
                    auto* src = src_offset(batch_index, row_id);
                    auto* dst = dst_offset(batch_index, row_id);
                    m_gemvImpl->exec(src, dst, wei, bias, scale, zp);
                    if (std::getenv("U3DBG") && m < 3 && m_nativeWeightsMemoryDbg) {
                        auto* dstF = static_cast<float*>(dst);
                        printf("[U3DBG dst] ge=%zu m=%d row_id=%zu dst[0..3]=%f,%f,%f,%f\n", gather_axis_index, m,
                               row_id, dstF[0], dstF[1], dstF[2], dstF[3]);
                        // Hand-compute a reference for oc=0,1 directly from the NATIVE (unrepacked,
                        // un-transposed) weight/scale/zp tensors and the ACTUAL src, bypassing
                        // oneDNN entirely -- to check whether the graph's own reference/interpreter
                        // computation (not the CPU plugin) is what disagrees.
                        const auto& weiDims = m_nativeWeightsMemoryDbg->getStaticDims();  // [G,OC,IC]
                        const auto OC = weiDims[1];
                        const auto Kd = weiDims[2];
                        const auto* srcF = static_cast<const float*>(src);
                        const auto* scaleF = m_nativeScalesMemoryDbg->getDataAs<float>();  // [G,OC,groups]
                        const auto& scDims = m_nativeScalesMemoryDbg->getStaticDims();
                        const auto Gg = scDims[2];
                        const size_t groupSize = Kd / Gg;
                        printf("[U3DBG shapes] weiDims=%zu,%zu,%zu scDims=%zu,%zu,%zu groupSize=%zu wei_prc=%s "
                               "zp_prc=%s zpMem_actual_prc=%s scaleMem_actual_prc=%s\n",
                               weiDims[0], weiDims[1], weiDims[2], scDims[0], scDims[1], scDims[2], groupSize,
                               m_weightsPrecisionDbg.get_type_name().c_str(), m_zpPrecisionDbg.get_type_name().c_str(),
                               m_nativeZpMemoryDbg ? m_nativeZpMemoryDbg->getDesc().getPrecision().get_type_name().c_str()
                                                    : "null",
                               m_nativeScalesMemoryDbg->getDesc().getPrecision().get_type_name().c_str());
                        // Use OpenVINO's own element::iterator directly (the exact mechanism the
                        // test builder used to WRITE this data) instead of a hand-rolled decode,
                        // to eliminate any chance of a hand-rolled bit-convention mistake.
                        auto decode_native = [](ov::element::Type prc, const uint8_t* base, size_t idx) -> uint8_t {
                            const auto* signedBase = reinterpret_cast<const int8_t*>(base);
                            if (prc == ov::element::u3) {
                                auto it = ov::element::iterator<ov::element::u3>(signedBase);
                                return static_cast<uint8_t>(static_cast<int8_t>(*(it + static_cast<std::ptrdiff_t>(idx))));
                            }
                            auto it = ov::element::iterator<ov::element::u4>(signedBase);
                            return static_cast<uint8_t>(static_cast<int8_t>(*(it + static_cast<std::ptrdiff_t>(idx))));
                        };
                        const auto* weiRaw = static_cast<const uint8_t*>(m_nativeWeightsMemoryDbg->getData());
                        const auto* zpRaw = m_nativeZpMemoryDbg ? static_cast<const uint8_t*>(m_nativeZpMemoryDbg->getData())
                                                                 : nullptr;
                        for (size_t oc = 0; oc < 2 && oc < OC; oc++) {
                            double ref = 0.0;
                            for (size_t ic = 0; ic < Kd; ic++) {
                                size_t widx = gather_axis_index * OC * Kd + oc * Kd + ic;
                                uint8_t wv = decode_native(m_weightsPrecisionDbg, weiRaw, widx);
                                size_t g = ic / groupSize;
                                float scaleV = scaleF[gather_axis_index * OC * Gg + oc * Gg + g];
                                uint8_t zpV = zpRaw ? decode_native(m_zpPrecisionDbg, zpRaw,
                                                                     gather_axis_index * OC * Gg + oc * Gg + g)
                                                     : 0;
                                ref += static_cast<double>(srcF[ic]) * (static_cast<double>(wv) - zpV) * scaleV;
                            }
                            printf("[U3DBG refcheck] ge=%zu m=%d oc=%zu ref=%f actual=%f\n", gather_axis_index, m, oc,
                                   ref, dstF[oc]);
                        }
                        fflush(stdout);
                    }
                }
            }
        }
    } else {
        OPENVINO_ASSERT(m_gemvImpl, "GEMV implementation is not created");

        constexpr size_t m = 0;
        auto* gather_ids = static_cast<int32_t*>(index_offset(m));
        for (size_t i = 0; i < indices_size; i++) {
            int32_t gather_axis_index = gather_ids[i];
            OPENVINO_ASSERT(gather_axis_index >= 0 && static_cast<size_t>(gather_axis_index) < gather_axis_size,
                            "Invalid gather_id ",
                            gather_axis_index,
                            " for i ",
                            i);
            auto* src = src_offset(i, m);
            auto* dst = dst_offset(i, m);
            auto* wei = wei_offset(gather_axis_index);
            auto* bias = bias_offset(gather_axis_index);
            auto* scale = scale_offset(gather_axis_index);
            auto* zp = zp_offset(gather_axis_index);
            m_gemvImpl->exec(src, dst, wei, bias, scale, zp);
            if (std::getenv("U3DBG")) {
                auto* dstF = static_cast<float*>(dst);
                printf("[U3DBG dst] ge=%d i=%zu dst[0..3]=%f,%f,%f,%f\n", gather_axis_index, i, dstF[0], dstF[1],
                       dstF[2], dstF[3]);
                fflush(stdout);
            }
        }
    }
}

impl_desc_type GatherMatmulDnnlExecutor::implType() const {
    return m_implType;
}

}  // namespace ov::intel_cpu
