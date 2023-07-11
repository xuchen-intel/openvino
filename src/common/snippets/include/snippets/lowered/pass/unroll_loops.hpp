// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"
#include "snippets/generator.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface UnrollLoops
 * @brief Unroll loops containing Eltwise nodes.
 * @ingroup snippets
 */
class UnrollLoops : public Pass {
public:
    OPENVINO_RTTI("Unroll_loops", "Pass")
    explicit UnrollLoops(const std::function<Generator::opRegType(const std::shared_ptr<Node>& op)>& mapper) : m_reg_type_mapper(mapper) {}
    bool run(LinearIR& linear_ir) override;

private:
    std::function<Generator::opRegType(const std::shared_ptr<Node>& op)> m_reg_type_mapper;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
