// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <string>

#include "openvino/util/file_util.hpp"
#include "primitive_inst.h"

namespace cldnn::ocl::debug_dump {

inline std::string root_with_separator(const char* dump_root_env) {
    std::string dump_root = dump_root_env;
    if (!dump_root.empty() && dump_root.back() != '/') {
        dump_root.push_back('/');
    }
    return dump_root;
}

inline std::string layout_suffix(const cldnn::layout& mem_layout) {
    std::string dims;
    const auto mem_dims = mem_layout.get_dims();
    for (size_t i = 0; i < mem_layout.get_rank(); i++) {
        dims += "_" + std::to_string(mem_dims[i]);
    }

    return ov::element::Type(mem_layout.data_type).get_type_name() + dims + "__" + mem_layout.format.to_string();
}

inline std::string sanitize_node_name(const std::string& node_name) {
    std::string sanitized;
    sanitized.reserve(node_name.size());
    for (const unsigned char ch : node_name) {
        if (std::isalnum(ch) || ch == '_' || ch == '-' || ch == '.') {
            sanitized.push_back(static_cast<char>(ch));
        } else {
            sanitized.push_back('_');
        }
    }
    return sanitized;
}

inline std::string trim_copy(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\n\r");
    return value.substr(first, last - first + 1);
}

inline bool matches_filter_list(const std::string& node_name, const char* filter_env) {
    if (filter_env == nullptr || filter_env[0] == '\0') {
        return false;
    }

    const std::string filters = filter_env;
    size_t start = 0;
    while (start <= filters.size()) {
        const auto end = filters.find(',', start);
        const auto token = trim_copy(filters.substr(start, end == std::string::npos ? std::string::npos : end - start));
        if (!token.empty() && node_name.find(token) != std::string::npos) {
            return true;
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }

    return false;
}

inline bool should_dump_selected_output(const cldnn::primitive_inst& instance) {
    const char* dump_root_env = std::getenv("DUMP_OCL_OUTPUTS_ROOT");
    const char* filter_env = std::getenv("DUMP_OCL_OUTPUTS_FILTER");
    if (dump_root_env == nullptr || dump_root_env[0] == '\0' || filter_env == nullptr || filter_env[0] == '\0') {
        return false;
    }

    return matches_filter_list(instance.get_node().id(), filter_env);
}

inline bool should_dump_selected_input(const cldnn::primitive_inst& instance) {
    const char* dump_root_env = std::getenv("DUMP_OCL_OUTPUTS_ROOT");
    const char* filter_env = std::getenv("DUMP_OCL_INPUTS_FILTER");
    if (dump_root_env == nullptr || dump_root_env[0] == '\0') {
        return false;
    }

    return matches_filter_list(instance.get_node().id(), filter_env);
}

inline void dump_memory(const std::string& root,
                        const char* impl_name,
                        size_t execute_idx,
                        const std::string& sanitized_node_name,
                        const std::string& tensor_name,
                        const cldnn::memory::ptr& mem,
                        cldnn::stream& stream) {
    if (mem == nullptr) {
        return;
    }

    cldnn::mem_lock<char, cldnn::mem_lock_type::read> lock(mem, stream);
    const auto& mem_layout = mem->get_layout();
    const auto path = root + impl_name + "__exec_" + std::to_string(execute_idx) + "__" + sanitized_node_name + "__" + tensor_name + "__" +
                      layout_suffix(mem_layout) + ".bin";
    ov::util::save_binary(path, lock.data(), mem->size());
}

inline void dump_dependency_info(const std::string& root,
                                 const char* impl_name,
                                 size_t execute_idx,
                                 const cldnn::primitive_inst& instance,
                                 const std::string& sanitized_node_name) {
    const auto path = root + impl_name + "__exec_" + std::to_string(execute_idx) + "__" + sanitized_node_name + "__deps.txt";
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        return;
    }

    out << "node=" << instance.id() << '\n';
    for (size_t input_idx = 0; input_idx < instance.dependencies().size(); input_idx++) {
        const auto& dep = instance.dependencies().at(input_idx);
        out << "input_" << input_idx << "=" << dep.first->id() << ":output_" << dep.second << '\n';
    }
}

inline void dump_selected_output(cldnn::primitive_inst& instance, const char* impl_name) {
    if (!should_dump_selected_output(instance)) {
        return;
    }

    const char* dump_root_env = std::getenv("DUMP_OCL_OUTPUTS_ROOT");
    OPENVINO_ASSERT(dump_root_env != nullptr);

    static std::atomic<size_t> execute_counter{0};
    const size_t execute_idx = execute_counter.fetch_add(1);
    auto& stream = instance.get_network().get_stream();
    stream.finish();

    try {
        const auto root = root_with_separator(dump_root_env);
        const auto sanitized_node_name = sanitize_node_name(instance.get_node().id());
        dump_dependency_info(root, impl_name, execute_idx, instance, sanitized_node_name);
        if (should_dump_selected_input(instance)) {
            for (size_t input_idx = 0; input_idx < instance.inputs_memory_count(); input_idx++) {
                const auto tensor_name = instance.inputs_memory_count() == 1 ? std::string("input")
                                                                             : "input_" + std::to_string(input_idx);
                dump_memory(root,
                            impl_name,
                            execute_idx,
                            sanitized_node_name,
                            tensor_name,
                            instance.input_memory_ptr(input_idx),
                            stream);
            }
        }

        for (size_t output_idx = 0; output_idx < instance.outputs_memory_count(); output_idx++) {
            const auto tensor_name = instance.outputs_memory_count() == 1 ? std::string("output")
                                                                           : "output_" + std::to_string(output_idx);
            dump_memory(root,
                        impl_name,
                        execute_idx,
                        sanitized_node_name,
                        tensor_name,
                        instance.output_memory_ptr(output_idx),
                        stream);
        }
    } catch (const std::exception& e) {
        std::cerr << "[dump_selected_output] Failed to dump GPU output for " << instance.get_node().id() << ": " << e.what() << "\n";
    }
}

}  // namespace cldnn::ocl::debug_dump