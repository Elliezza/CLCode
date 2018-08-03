/*
 * Copyright (c) 2017-2018 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/graph.h"
#include "support/ToolchainSupport.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <memory>
#include <chrono>
#include <unistd.h>

using namespace arm_compute;
using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

/** Example demonstrating how to implement QASYMM8 MobileNet's network using the Compute Library's graph API
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Target (0 = NEON, 1 = OpenCL, 2 = OpenCL with Tuner), [optional] Path to the weights folder, [optional] npy_input, [optional] labels, [optional] Fast math for convolution layer (0 = DISABLED, 1 = ENABLED) )
 */
class GraphMobileNetQASYMM8Example : public Example
{
public:
    void do_setup(int argc, char **argv) override
    {
        std::string data_path; /* Path to the trainable data */
        std::string input;     /* Image data */
        std::string label;     /* Label data */

        // Quantization info taken from the AndroidNN QASYMM8 MobileNet example
        const QuantizationInfo in_quant_info  = QuantizationInfo(0.0078125f, 128);
        const QuantizationInfo mid_quant_info = QuantizationInfo(0.0784313753247f, 128);

        const std::vector<QuantizationInfo> conv_weights_quant_info =
        {
            QuantizationInfo(0.031778190285f, 156), // conv0
            QuantizationInfo(0.00604454148561f, 66) // conv14
        };

        const std::vector<QuantizationInfo> depth_weights_quant_info =
        {
            QuantizationInfo(0.254282623529f, 129),  // dwsc1
            QuantizationInfo(0.12828284502f, 172),   // dwsc2
            QuantizationInfo(0.265911251307f, 83),   // dwsc3
            QuantizationInfo(0.0985597148538f, 30),  // dwsc4
            QuantizationInfo(0.0631204470992f, 54),  // dwsc5
            QuantizationInfo(0.0137207424268f, 141), // dwsc6
            QuantizationInfo(0.0817828401923f, 125), // dwsc7
            QuantizationInfo(0.0393880493939f, 164), // dwsc8
            QuantizationInfo(0.211694166064f, 129),  // dwsc9
            QuantizationInfo(0.158015936613f, 103),  // dwsc10
            QuantizationInfo(0.0182712618262f, 137), // dwsc11
            QuantizationInfo(0.0127998134121f, 134), // dwsc12
            QuantizationInfo(0.299285322428f, 161)   // dwsc13
        };

        const std::vector<QuantizationInfo> point_weights_quant_info =
        {
            QuantizationInfo(0.0425766184926f, 129),  // dwsc1
            QuantizationInfo(0.0250773020089f, 94),   // dwsc2
            QuantizationInfo(0.015851572156f, 93),    // dwsc3
            QuantizationInfo(0.0167811904103f, 98),   // dwsc4
            QuantizationInfo(0.00951790809631f, 135), // dwsc5
            QuantizationInfo(0.00999817531556f, 128), // dwsc6
            QuantizationInfo(0.00590536883101f, 126), // dwsc7
            QuantizationInfo(0.00576109671965f, 133), // dwsc8
            QuantizationInfo(0.00830461271107f, 142), // dwsc9
            QuantizationInfo(0.0152327232063f, 72),   // dwsc10
            QuantizationInfo(0.00741417845711f, 125), // dwsc11
            QuantizationInfo(0.0135628981516f, 142),  // dwsc12
            QuantizationInfo(0.0338749065995f, 140)   // dwsc13
        };

        // Set target. 0 (NEON), 1 (OpenCL), 2 (OpenCL with Tuner). By default it is NEON
        const int    target         = argc > 1 ? std::strtol(argv[1], nullptr, 10) : 0;
        Target       target_hint    = set_target_hint(target);
        FastMathHint fast_math_hint = FastMathHint::DISABLED;

        // Parse arguments
        if(argc < 2)
        {
            // Print help
            std::cout << "Usage: " << argv[0] << " [target] [path_to_data] [npy_input] [labels] [fast_math_hint]\n\n";
            std::cout << "No data folder provided: using random values\n\n";
        }
        else if(argc == 2)
        {
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " [path_to_data] [npy_input] [labels] [fast_math_hint]\n\n";
            std::cout << "No input provided: using random values\n\n";
        }
        else if(argc == 4)
        {
            data_path = argv[2];
            input     = argv[3];
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " " << argv[2] << " " << argv[3] << " [labels] [fast_math_hint]\n\n";
            std::cout << "No text file with labels provided: skipping output accessor\n\n";
        }
        else if(argc == 5)
        {
            data_path = argv[2];
            input     = argv[3];
            label     = argv[4];
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " " << argv[2] << " " << argv[3] << " " << argv[4] << " [fast_math_hint]\n\n";
            std::cout << "No fast math info provided: disabling fast math\n\n";
        }
        else
        {
            data_path      = argv[2];
            input          = argv[3];
            label          = argv[4];
            fast_math_hint = (std::strtol(argv[5], nullptr, 1) == 0) ? FastMathHint::DISABLED : FastMathHint::ENABLED;
        }

        graph << target_hint
              << DepthwiseConvolutionMethod::OPTIMIZED_3x3
              << fast_math_hint
              << InputLayer(TensorDescriptor(TensorShape(224U, 224U, 3U, 1U), DataType::QASYMM8, in_quant_info),
                            get_weights_accessor(data_path, "/cnn_data/mobilenet_qasymm8_model/" + input))
              << ConvolutionLayer(
                  3U, 3U, 32U,
                  get_weights_accessor(data_path, "/cnn_data/mobilenet_qasymm8_model/Conv2d_0_weights.npy"),
                  get_weights_accessor(data_path, "/cnn_data/mobilenet_qasymm8_model/Conv2d_0_bias.npy"),
                  PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR),
                  1, conv_weights_quant_info.at(0), mid_quant_info)
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f));
        graph << get_dwsc_node(data_path, "Conv2d_1", 64U, PadStrideInfo(1U, 1U, 1U, 1U), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(0), point_weights_quant_info.at(0));
        graph << get_dwsc_node(data_path, "Conv2d_2", 128U, PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(1),
                               point_weights_quant_info.at(1));
        graph << get_dwsc_node(data_path, "Conv2d_3", 128U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(2),
                               point_weights_quant_info.at(2));
        graph << get_dwsc_node(data_path, "Conv2d_4", 256U, PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(3),
                               point_weights_quant_info.at(3));
        graph << get_dwsc_node(data_path, "Conv2d_5", 256U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(4),
                               point_weights_quant_info.at(4));
        graph << get_dwsc_node(data_path, "Conv2d_6", 512U, PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(5),
                               point_weights_quant_info.at(5));
        graph << get_dwsc_node(data_path, "Conv2d_7", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(6),
                               point_weights_quant_info.at(6));
        graph << get_dwsc_node(data_path, "Conv2d_8", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(7),
                               point_weights_quant_info.at(7));
        graph << get_dwsc_node(data_path, "Conv2d_9", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(8),
                               point_weights_quant_info.at(8));
        graph << get_dwsc_node(data_path, "Conv2d_10", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(9),
                               point_weights_quant_info.at(9));
        graph << get_dwsc_node(data_path, "Conv2d_11", 512U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(10),
                               point_weights_quant_info.at(10));
        graph << get_dwsc_node(data_path, "Conv2d_12", 1024U, PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(11),
                               point_weights_quant_info.at(11));
        graph << get_dwsc_node(data_path, "Conv2d_13", 1024U, PadStrideInfo(1U, 1U, 1U, 1U, 1U, 1U, DimensionRoundingType::FLOOR), PadStrideInfo(1U, 1U, 0U, 0U), depth_weights_quant_info.at(12),
                               point_weights_quant_info.at(12))
              << PoolingLayer(PoolingLayerInfo(PoolingType::AVG))
              << ConvolutionLayer(
                  1U, 1U, 1001U,
                  get_weights_accessor(data_path, "/cnn_data/mobilenet_qasymm8_model/Logits_Conv2d_1c_1x1_weights.npy"),
                  get_weights_accessor(data_path, "/cnn_data/mobilenet_qasymm8_model/Logits_Conv2d_1c_1x1_bias.npy"),
                  PadStrideInfo(1U, 1U, 0U, 0U), 1, conv_weights_quant_info.at(1))
              << ReshapeLayer(TensorShape(1001U))
              << SoftmaxLayer()
              << OutputLayer(get_output_accessor(label, 5));

        // Finalize graph
        GraphConfig config;
        config.use_tuner = (target == 2);
        graph.finalize(target_hint, config);
    }
    void do_run() override
    {
        // Run graph
       auto tbegin = std::chrono::high_resolution_clock::now();
       for(int i=0; i<50; i++){
        graph.run();
       }
       auto tend = std::chrono::high_resolution_clock::now();
       double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count();
       double cost = cost0/50;

       std::cout << "COST:" << cost << std::endl;
    }

private:
    Stream graph{ 0, "MobileNetV1_QASYMM8" };

    /** This function produces a depthwise separable convolution node (i.e. depthwise + pointwise layers) with ReLU6 activation after each layer.
     *
     * @param[in] data_path                Path to trainable data folder
     * @param[in] param_path               Prefix of specific set of weights/biases data
     * @param[in] conv_filt                Filters depths for pointwise convolution
     * @param[in] dwc_pad_stride_info      PadStrideInfo for depthwise convolution
     * @param[in] conv_pad_stride_info     PadStrideInfo for pointwise convolution
     * @param[in] depth_weights_quant_info QuantizationInfo for depthwise convolution's weights
     * @param[in] point_weights_quant_info QuantizationInfo for pointwise convolution's weights
     *
     * @return The complete dwsc node
     */
    BranchLayer get_dwsc_node(const std::string &data_path, std::string &&param_path,
                              const unsigned int conv_filt,
                              PadStrideInfo dwc_pad_stride_info, PadStrideInfo conv_pad_stride_info,
                              QuantizationInfo depth_weights_quant_info, QuantizationInfo point_weights_quant_info)
    {
        std::string total_path = "/cnn_data/mobilenet_qasymm8_model/" + param_path + "_";
        SubStream   sg(graph);

        sg << DepthwiseConvolutionLayer(
               3U, 3U,
               get_weights_accessor(data_path, total_path + "depthwise_weights.npy"),
               get_weights_accessor(data_path, total_path + "depthwise_bias.npy"),
               dwc_pad_stride_info, depth_weights_quant_info)
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f))
           << ConvolutionLayer(
               1U, 1U, conv_filt,
               get_weights_accessor(data_path, total_path + "pointwise_weights.npy"),
               get_weights_accessor(data_path, total_path + "pointwise_bias.npy"),
               conv_pad_stride_info, 1, point_weights_quant_info)
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.f));

        return BranchLayer(std::move(sg));
    }
};
/** Main program for MobileNetQASYMM8
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Path to the weights folder, [optional] npy_input, [optional] labels, [optional] Fast math for convolution layer (0 = DISABLED, 1 = ENABLED) )
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphMobileNetQASYMM8Example>(argc, argv);
}
