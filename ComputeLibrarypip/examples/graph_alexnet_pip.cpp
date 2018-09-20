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
#include <sched.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <memory>

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

/** Example demonstrating how to implement AlexNet's network using the Compute Library's graph API
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Target (0 = NEON, 1 = OpenCL, 2 = OpenCL with Tuner), [optional] Path to the weights folder, [optional] image, [optional] labels, [optional] Fast math for convolution layer (0 = DISABLED, 1 = ENABLED) )
 */
class GraphAlexnetExample : public Example
{
public:
    void do_setup(int argc, char **argv) override
    {
        std::string data_path; /* Path to the trainable data */
        std::string image;     /* Image data */
        std::string label;     /* Label data */

//        // Create a preprocessor object
//        const std::array<float, 3> mean_rgb{ { 122.68f, 116.67f, 104.01f } };
//        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb);

        // Set target. 0 (NEON), 1 (OpenCL), 2 (OpenCL with Tuner). By default it is NEON
        const int target      = argc > 1 ? std::strtol(argv[1], nullptr, 10) : 0;
        Target    target_hint = set_target_hint(target);

        const bool        is_neon              = (target_hint == Target::NEON);
        ConvolutionMethod convolution_5x5_hint = is_neon ? ConvolutionMethod::GEMM : ConvolutionMethod::DIRECT;
        ConvolutionMethod convolution_3x3_hint = ConvolutionMethod::DEFAULT;
        FastMathHint      fast_math_hint       = FastMathHint::DISABLED;

        // Parse arguments
        if(argc < 2)
        {
            // Print help
            std::cout << "Usage: " << argv[0] << " [target] [path_to_data] [image] [labels] [fast_math_hint]\n\n";
            std::cout << "No data folder provided: using random values\n\n";
        }
        else if(argc == 2)
        {
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " [path_to_data] [image] [labels] [fast_math_hint]\n\n";
            std::cout << "No data folder provided: using random values\n\n";
        }
        else if(argc == 3)
        {
            data_path = argv[2];
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " " << argv[2] << " [image] [labels] [fast_math_hint]\n\n";
            std::cout << "No image provided: using random values\n\n";
        }
        else if(argc == 4)
        {
            data_path = argv[2];
            image     = argv[3];
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " " << argv[2] << " " << argv[3] << " [labels] [fast_math_hint]\n\n";
            std::cout << "No text file with labels provided: skipping output accessor\n\n";
        }
        else if(argc == 5)
        {
            data_path = argv[2];
            image     = argv[3];
            label     = argv[4];
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " " << argv[2] << " " << argv[3] << " " << argv[4] << " [fast_math_hint]\n\n";
            std::cout << "No fast math info provided: disabling fast math\n\n";
        }
        else
        {
            data_path      = argv[2];
            image          = argv[3];
            label          = argv[4];
            fast_math_hint = (std::strtol(argv[5], nullptr, 1) == 0) ? FastMathHint::DISABLED : FastMathHint::ENABLED;
        }

/*	graph << target_hint
              << fast_math_hint
	      << InputLayer(TensorDescriptor(TensorShape(227U, 227U, 3U, 1U), DataType::F32),
                            get_input_accessor(image, std::move(preprocessor)))
              // Layer 1
              << ConvolutionLayer(
                  11U, 11U, 96U,
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv1_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv1_b.npy"),
                  PadStrideInfo(4, 4, 0, 0))
              .set_name("conv1")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu1")
              << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f)).set_name("norm1")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0))).set_name("pool1")
              // Layer 2
              << convolution_5x5_hint
              << ConvolutionLayer(
                  5U, 5U, 256U,
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv2_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv2_b.npy"),
                  PadStrideInfo(1, 1, 2, 2), 2)
              .set_name("conv2")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu2")
              << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f)).set_name("norm2")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0))).set_name("pool2")
              << convolution_3x3_hint
              // Layer 3
              << ConvolutionLayer(
                  3U, 3U, 384U,
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv3_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv3_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              .set_name("conv3")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3")
              // Layer 4
              << ConvolutionLayer(
                  3U, 3U, 384U,
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv4_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv4_b.npy"),
                  PadStrideInfo(1, 1, 1, 1), 2)
              .set_name("conv4")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4")
              // Layer 5
              << ConvolutionLayer(
                  3U, 3U, 256U,
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv5_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv5_b.npy"),
                  PadStrideInfo(1, 1, 1, 1), 2)
              .set_name("conv5")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5")
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0))).set_name("pool5")
              // Layer 6
              << FullyConnectedLayer(
                  4096U,
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc6_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc6_b.npy"))
              .set_name("fc6")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu6")
              // Layer 7
              << FullyConnectedLayer(
                  4096U,
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc7_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc7_b.npy"))
              .set_name("fc7")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu7")
              // Layer 8
              << FullyConnectedLayer(
                  1000U,
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc8_w.npy"),
                  get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc8_b.npy"))
              .set_name("fc8")
              // Softmax
              << SoftmaxLayer().set_name("prob")
              << OutputLayer(get_output_accessor(label, 5));
*/

	initialize_graph(target_hint, data_path, image, label, fast_math_hint, convolution_5x5_hint, convolution_3x3_hint, graph0);
	initialize_graph(target_hint, data_path, image, label, fast_math_hint, convolution_5x5_hint, convolution_3x3_hint, graph1);
	initialize_graph(target_hint, data_path, image, label, fast_math_hint, convolution_5x5_hint, convolution_3x3_hint, graph2);
	initialize_graph(target_hint, data_path, image, label, fast_math_hint, convolution_5x5_hint, convolution_3x3_hint, graph3);
	initialize_graph(target_hint, data_path, image, label, fast_math_hint, convolution_5x5_hint, convolution_3x3_hint, graph4);
	initialize_graph(target_hint, data_path, image, label, fast_math_hint, convolution_5x5_hint, convolution_3x3_hint, graph5);
	initialize_graph(target_hint, data_path, image, label, fast_math_hint, convolution_5x5_hint, convolution_3x3_hint, graph6);
	initialize_graph(target_hint, data_path, image, label, fast_math_hint, convolution_5x5_hint, convolution_3x3_hint, graph7);

        // Finalize graph
        GraphConfig config;
        config.use_tuner = (target == 2);
        graph0.finalize(target_hint, config);
	graph1.finalize(target_hint, config);
/*	graph2.finalize(target_hint, config);
	graph3.finalize(target_hint, config);
	graph4.finalize(target_hint, config);
	graph5.finalize(target_hint, config);
	graph6.finalize(target_hint, config);
	graph7.finalize(target_hint, config);*/

    }
    
/*    void do_run() override
    {
        // Run graph
	    std::cout << "Starting of running the kernel" << std::endl;
	auto tbegin = std::chrono::high_resolution_clock::now();
       for(int i=0; i<1; i++){
        graph0.run();
       }
       auto tend = std::chrono::high_resolution_clock::now();
       double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count();
       double cost = cost0/1;
       //double cost = cost0;

       std::cout << "COST:" << cost << std::endl;
    }*/

    void do_run() override
    {
	    std::cout << "Start running of the graph" << std::endl;
	    int num_cores = 2;
	    auto tbegin = std::chrono::high_resolution_clock::now();

	    int k = 4;
	    std::vector<std::thread> workers(num_cores);
	    for(int i = 0; i < num_cores; ++i){
		    workers[i] = std::thread([&, i]{
				    {
				    std::cout << "Creating new threads: " << i << " on CPU:" << sched_getcpu() << std::endl;
				    if (i == 0) {
				    for (int j = 0; j < 10; j++) graph0.run();
				    } else if (i == 1){
				    for (int j = 0; j < 10; j++) graph1.run();
				    } /*else if (i == 2){
				    for (int j = 0; j < 10; j++) graph2.run();
				    } else if (i == 3){
				    for (int j = 0; j < 10; j++) graph3.run();
				    } else if (i == 4){
				    for (int j = 0; j < 10; j++) graph4.run();
				    } else if (i == 5){
				    for (int j = 0; j < 10; j++) graph5.run();
				    } else if (i == 6){
				    for (int j = 0; j < 10; j++) graph6.run();
				    } else if (i == 7){
				    for (int j = 0; j < 10; j++) graph7.run();}*/
				    }});
		    cpu_set_t cpuset;
		    CPU_ZERO(&cpuset);
		    CPU_SET((k+i/2), &cpuset);
		    int rc= pthread_setaffinity_np(workers[i].native_handle(), sizeof(cpu_set_t), &cpuset);
		    if (rc !=0) std::cout << "Error in setting affinity for thread " << i << std::endl;
	    }
	    for(auto&t: workers) t.join();
	    auto tend = std::chrono::high_resolution_clock::now();
	    double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count();
	    double cost = cost0/20;
	    std::cout << "COST:" << cost << std::endl;
    }


private:
    Stream graph0{ 0, "AlexNet" };
    Stream graph1{ 1, "AlexNet" };
    Stream graph2{ 2, "AlexNet" };
    Stream graph3{ 3, "AlexNet" };
    Stream graph4{ 4, "AlexNet" };
    Stream graph5{ 5, "AlexNet" };
    Stream graph6{ 6, "AlexNet" };
    Stream graph7{ 7, "AlexNet" };



    void initialize_graph(Target &target_hint, const std::string &data_path, const std::string &image, const std::string &label, FastMathHint &fast_math_hint, ConvolutionMethod &convolution_5x5_hint, ConvolutionMethod &convolution_3x3_hint, Stream &graph)
    {
	    // Create a preprocessor object
	    const std::array<float, 3> mean_rgb{ { 122.68f, 116.67f, 104.01f } };
	    std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb);

	    graph << target_hint
		    << fast_math_hint
		    << InputLayer(TensorDescriptor(TensorShape(227U, 227U, 3U, 1U), DataType::F32),
				    get_input_accessor(image, std::move(preprocessor)))
 // Layer 1
		    << ConvolutionLayer(
				    11U, 11U, 96U,
				    get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv1_w.npy"),
				    get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv1_b.npy"),
				    PadStrideInfo(4, 4, 0, 0))
		    .set_name("conv1")
		    << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu1")
		    << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f)).set_name("norm1")
		    << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0))).set_name("pool1")
 // Layer 2
		    << convolution_5x5_hint
		    << ConvolutionLayer(
				    5U, 5U, 256U,
				    get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv2_w.npy"),
				    get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv2_b.npy"),
				    PadStrideInfo(1, 1, 2, 2), 2)
		    .set_name("conv2")
		    << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu2")
		    << NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f)).set_name("norm2")
		    << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0))).set_name("pool2")
		    << convolution_3x3_hint
 // Layer 3
		    << ConvolutionLayer(
				    3U, 3U, 384U,
				    get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv3_w.npy"),
				    get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv3_b.npy"),
				    PadStrideInfo(1, 1, 1, 1))
		    .set_name("conv3")
		    << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu3")
 // Layer 4
		    << ConvolutionLayer(
				    3U, 3U, 384U,
				    get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv4_w.npy"),
				    get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv4_b.npy"),
				    PadStrideInfo(1, 1, 1, 1), 2)
		    .set_name("conv4")
		    << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu4")
 // Layer 5
		    << ConvolutionLayer(
				    3U, 3U, 256U,
				    get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv5_w.npy"),
				    get_weights_accessor(data_path, "/cnn_data/alexnet_model/conv5_b.npy"),
				    PadStrideInfo(1, 1, 1, 1), 2)
		    .set_name("conv5")
		    << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu5")
		    << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0))).set_name("pool5")
 // Layer 6
		    << FullyConnectedLayer(
				    4096U,
				    get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc6_w.npy"),
				    get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc6_b.npy"))
		    .set_name("fc6")
		    << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu6")
 // Layer 7
		    << FullyConnectedLayer(
				    4096U,
				    get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc7_w.npy"),
				    get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc7_b.npy"))
		    .set_name("fc7")
		    << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("relu7")
 // Layer 8
		    << FullyConnectedLayer(
				    1000U,
				    get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc8_w.npy"),
				    get_weights_accessor(data_path, "/cnn_data/alexnet_model/fc8_b.npy"))
		    .set_name("fc8")
// Softmax
		    << SoftmaxLayer().set_name("prob")
		    << OutputLayer(get_output_accessor(label, 5));
		
    }

};

/** Main program for AlexNet
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Target (0 = NEON, 1 = OpenCL, 2 = OpenCL with Tuner), [optional] Path to the weights folder, [optional] image, [optional] labels, [optional] Fast math for convolution layer (0 = DISABLED, 1 = ENABLED) )
 */
int main(int argc, char **argv)
{
	/*cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(4, &cpuset);
	int e = sched_setaffinity(getpid(), sizeof(cpuset), &cpuset);
	if(e !=0) {
		std::cout << "Error in setting sched_setaffinity \n";
	}*/
    return arm_compute::utils::run_example<GraphAlexnetExample>(argc, argv);
}
