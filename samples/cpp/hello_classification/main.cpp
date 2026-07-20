// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/classification_results.h"
#include "samples/slog.hpp"
#include "format_reader_ptr.h"
// clang-format on

/**
 * @brief Main with support Unicode paths, wide strings
 *
 * Usage modes:
 *   1) Normal classification:
 *        hello_classification.exe <model.onnx> <image.bmp> <device>
 *
 *   2) Export blob (compile model, save to .blob):
 *        hello_classification.exe export <model.onnx> <output.blob> <device>
 *
 *   3) Import blob (load .blob, run inference):
 *        hello_classification.exe import <model.blob> <device>
 */
int tmain(int argc, tchar* argv[]) {
    try {
        // -------- Set OpenVINO log message capturing callback --------
        const std::function<void(std::string_view)> log_callback{[](std::string_view msg) {
            slog::info << msg;
        }};
        ov::util::set_log_callback(log_callback);

        // -------- Get OpenVINO runtime version --------
        slog::info << ov::get_openvino_version() << slog::endl;

        // -------- Determine mode --------
        std::string mode = "";
        if (argc >= 2) {
            mode = TSTRING2STRING(argv[1]);
        }

        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        ov::Core core;

        if (mode == "export") {
            // ================================================================
            // EXPORT MODE: Read ONNX/XML → compile to device → export as .blob
            // Usage: hello_classification.exe export <model.onnx> <output.blob> <device>
            // ================================================================
            if (argc != 5) {
                slog::info << "Export usage: " << TSTRING2STRING(argv[0])
                           << " export <model.onnx> <output.blob> <device>" << slog::endl;
                slog::info << "  Example: hello_classification.exe export model.onnx model_cpu.blob CPU"
                           << slog::endl;
                return EXIT_FAILURE;
            }

            const std::string model_path = TSTRING2STRING(argv[2]);
            const std::string blob_path = TSTRING2STRING(argv[3]);
            const std::string device_name = TSTRING2STRING(argv[4]);

            // Read model
            slog::info << "Reading model: " << model_path << slog::endl;
            auto model = core.read_model(model_path);
            slog::info << "Model inputs/outputs:" << slog::endl;
            printInputAndOutputsInfoShort(*model);

            // Compile model to device
            slog::info << "Compiling model to device: " << device_name << slog::endl;
            auto compiled_model = core.compile_model(model, device_name);
            slog::info << "Compilation successful!" << slog::endl;

            auto exec_devices = compiled_model.get_property(ov::execution_devices);
            slog::info << "Execution devices: " << exec_devices << slog::endl;

            // Export to blob
            std::ofstream blob_file(blob_path, std::ios::binary);
            if (!blob_file.is_open()) {
                throw std::logic_error("Cannot open output blob file: " + blob_path);
            }
            compiled_model.export_model(blob_file);
            blob_file.close();
            slog::info << "Model exported to blob: " << blob_path << slog::endl;

        } else if (mode == "import") {
            // ================================================================
            // IMPORT MODE: Load .blob → import_model → run inference
            // Usage: hello_classification.exe import <model.blob> <device>
            // ================================================================
            if (argc != 4) {
                slog::info << "Import usage: " << TSTRING2STRING(argv[0])
                           << " import <model.blob> <device>" << slog::endl;
                slog::info << "  Example: hello_classification.exe import model_cpu.blob CPU"
                           << slog::endl;
                return EXIT_FAILURE;
            }

            const std::string blob_path = TSTRING2STRING(argv[2]);
            const std::string device_name = TSTRING2STRING(argv[3]);

            // Import blob
            slog::info << "Importing compiled model (blob): " << blob_path << slog::endl;

            std::ifstream blob_file(blob_path, std::ios::binary);
            if (!blob_file.is_open()) {
                throw std::logic_error("Cannot open blob file: " + blob_path);
            }

            ov::CompiledModel compiled_model = core.import_model(blob_file, device_name);
            blob_file.close();

            slog::info << "Import successful!" << slog::endl;

            auto exec_devices = compiled_model.get_property(ov::execution_devices);
            slog::info << "Execution devices: " << exec_devices << slog::endl;

            slog::info << "Model inputs/outputs:" << slog::endl;
            printInputAndOutputsInfoShort(compiled_model);

            // Create infer request and run inference with zero-filled inputs
            ov::InferRequest infer_request = compiled_model.create_infer_request();

            for (size_t i = 0; i < compiled_model.inputs().size(); i++) {
                auto input_tensor = infer_request.get_input_tensor(i);
                memset(input_tensor.data(), 0, input_tensor.get_byte_size());
            }

            slog::info << "Running inference..." << slog::endl;
            infer_request.infer();
            slog::info << "Inference on imported blob completed successfully!" << slog::endl;

        } else {
            // ================================================================
            // NORMAL MODE: Original classification flow
            // Usage: hello_classification.exe <model.onnx> <image.bmp> <device>
            // ================================================================
            if (argc != 4) {
                slog::info << "Usage:" << slog::endl;
                slog::info << "  Classification: " << TSTRING2STRING(argv[0])
                           << " <model.onnx> <image.bmp> <device>" << slog::endl;
                slog::info << "  Export blob:    " << TSTRING2STRING(argv[0])
                           << " export <model.onnx> <output.blob> <device>" << slog::endl;
                slog::info << "  Import blob:    " << TSTRING2STRING(argv[0])
                           << " import <model.blob> <device>" << slog::endl;
                return EXIT_FAILURE;
            }

            const std::string model_path = TSTRING2STRING(argv[1]);
            const std::string image_path = TSTRING2STRING(argv[2]);
            const std::string device_name = TSTRING2STRING(argv[3]);

            // -------- Step 2. Read a model --------
            slog::info << "Loading model files: " << model_path << slog::endl;
            std::shared_ptr<ov::Model> model = core.read_model(model_path);
            printInputAndOutputsInfo(*model);

            OPENVINO_ASSERT(model->inputs().size() == 1, "Sample supports models with 1 input only");
            OPENVINO_ASSERT(model->outputs().size() == 1, "Sample supports models with 1 output only");

            // -------- Step 3. Set up input
            FormatReader::ReaderPtr reader(image_path.c_str());
            if (reader.get() == nullptr) {
                std::stringstream ss;
                ss << "Image " + image_path + " cannot be read!";
                throw std::logic_error(ss.str());
            }

            ov::element::Type input_type = ov::element::u8;
            ov::Shape input_shape = {1, reader->height(), reader->width(), 3};
            std::shared_ptr<unsigned char> input_data = reader->getData();

            ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, input_data.get());
            const ov::Layout tensor_layout{"NHWC"};

            // -------- Step 4. Configure preprocessing --------
            ov::preprocess::PrePostProcessor ppp(model);
            ppp.input().tensor().set_shape(input_shape).set_element_type(input_type).set_layout(tensor_layout);
            ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
            ppp.input().model().set_layout("NCHW");
            ppp.output().tensor().set_element_type(ov::element::f32);
            model = ppp.build();

            // -------- Step 5. Loading a model to the device --------
            ov::CompiledModel compiled_model = core.compile_model(model, device_name);

            // -------- Step 6. Create an infer request --------
            ov::InferRequest infer_request = compiled_model.create_infer_request();

            // -------- Step 7. Prepare input --------
            infer_request.set_input_tensor(input_tensor);

            // -------- Step 8. Do inference synchronously --------
            infer_request.infer();

            // -------- Step 9. Process output
            const ov::Tensor& output_tensor = infer_request.get_output_tensor();

            // Print classification results
            ClassificationResult classification_result(output_tensor, {image_path});
            classification_result.show();
        }
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}