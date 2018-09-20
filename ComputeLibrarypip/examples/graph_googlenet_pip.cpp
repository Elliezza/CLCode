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
#include <tuple>

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

/** Example demonstrating how to implement Googlenet's network using the Compute Library's graph API
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Target (0 = NEON, 1 = OpenCL, 2 = OpenCL with Tuner), [optional] Path to the weights folder, [optional] image, [optional] labels, [optional] Fast math for convolution layer (0 = DISABLED, 1 = ENABLED) )
 */
class GraphGooglenetExample : public Example
{
public:
    void do_setup(int argc, char **argv) override
    {
        std::string data_path; /* Path to the trainable data */
        std::string image;     /* Image data */
        std::string label;     /* Label data */

        // Create a preprocessor object
  //      const std::array<float, 3> mean_rgb{ { 122.68f, 116.67f, 104.01f } };
  //      std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb);

        // Set target. 0 (NEON), 1 (OpenCL), 2 (OpenCL with Tuner). By default it is NEON
        const int    target         = argc > 1 ? std::strtol(argv[1], nullptr, 10) : 0;
        Target       target_hint    = set_target_hint(target);
        FastMathHint fast_math_hint = FastMathHint::DISABLED;

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

        initialize_graph(target_hint, data_path, image, label, fast_math_hint, graph0);
        initialize_graph(target_hint, data_path, image, label, fast_math_hint, graph1);
        initialize_graph(target_hint, data_path, image, label, fast_math_hint, graph2);
        initialize_graph(target_hint, data_path, image, label, fast_math_hint, graph3);
        initialize_graph(target_hint, data_path, image, label, fast_math_hint, graph4);
        initialize_graph(target_hint, data_path, image, label, fast_math_hint, graph5);
        initialize_graph(target_hint, data_path, image, label, fast_math_hint, graph6);
        initialize_graph(target_hint, data_path, image, label, fast_math_hint, graph7);	

	// Finalize graph
        GraphConfig config;
        config.use_tuner = (target == 2);
	graph0.finalize(target_hint, config);
	graph1.finalize(target_hint, config);
	/*graph2.finalize(target_hint, config);
	graph3.finalize(target_hint, config);
	graph4.finalize(target_hint, config);
	graph5.finalize(target_hint, config);
	graph6.finalize(target_hint, config);
	graph7.finalize(target_hint, config);*/

    }

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
    Stream graph0{ 0, "GoogleNet" };
    Stream graph1{ 1, "GoogleNet" };
    Stream graph2{ 2, "GoogleNet" };
    Stream graph3{ 3, "GoogleNet" };
    Stream graph4{ 4, "GoogleNet" };
    Stream graph5{ 5, "GoogleNet" };
    Stream graph6{ 6, "GoogleNet" };
    Stream graph7{ 7, "GoogleNet" };

    void initialize_graph(Target &target_hint, const std::string &data_path, const std::string &image, const std::string &label, FastMathHint &fast_math_hint, Stream &graph)
    {
       // Create a preprocessor object
        const std::array<float, 3> mean_rgb{ { 122.68f, 116.67f, 104.01f } };
	std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb);

	graph << target_hint
		<< fast_math_hint
		<< InputLayer(TensorDescriptor(TensorShape(224U, 224U, 3U, 1U), DataType::F32),
				get_input_accessor(image, std::move(preprocessor)))
		<< ConvolutionLayer(
				7U, 7U, 64U,
				get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv1/conv1_7x7_s2_w.npy"),
				get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv1/conv1_7x7_s2_b.npy"),
				PadStrideInfo(2, 2, 3, 3))
		<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
		<< PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
		<< NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f))
		<< ConvolutionLayer(
				1U, 1U, 64U,
				get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_reduce_w.npy"),
				get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_reduce_b.npy"),
				PadStrideInfo(1, 1, 0, 0))
		<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
		<< ConvolutionLayer(
				3U, 3U, 192U,
				get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_w.npy"),
				get_weights_accessor(data_path, "/cnn_data/googlenet_model/conv2/conv2_3x3_b.npy"),
				PadStrideInfo(1, 1, 1, 1))
		<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
		<< NormalizationLayer(NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f))
		<< PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)));
	graph << get_inception_node(data_path, "inception_3a", 64, std::make_tuple(96U, 128U), std::make_tuple(16U, 32U), 32U, graph);
	graph << get_inception_node(data_path, "inception_3b", 128, std::make_tuple(128U, 192U), std::make_tuple(32U, 96U), 64U, graph);
	graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)));
	graph << get_inception_node(data_path, "inception_4a", 192, std::make_tuple(96U, 208U), std::make_tuple(16U, 48U), 64U, graph);
	graph << get_inception_node(data_path, "inception_4b", 160, std::make_tuple(112U, 224U), std::make_tuple(24U, 64U), 64U, graph);
	graph << get_inception_node(data_path, "inception_4c", 128, std::make_tuple(128U, 256U), std::make_tuple(24U, 64U), 64U, graph);
	graph << get_inception_node(data_path, "inception_4d", 112, std::make_tuple(144U, 288U), std::make_tuple(32U, 64U), 64U, graph);
	graph << get_inception_node(data_path, "inception_4e", 256, std::make_tuple(160U, 320U), std::make_tuple(32U, 128U), 128U, graph);
	graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)));
	graph << get_inception_node(data_path, "inception_5a", 256, std::make_tuple(160U, 320U), std::make_tuple(32U, 128U), 128U, graph);
	graph << get_inception_node(data_path, "inception_5b", 384, std::make_tuple(192U, 384U), std::make_tuple(48U, 128U), 128U, graph);
	graph << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 7, PadStrideInfo(1, 1, 0, 0, DimensionRoundingType::CEIL)))
		<< FullyConnectedLayer(
				1000U,
				get_weights_accessor(data_path, "/cnn_data/googlenet_model/loss3/loss3_classifier_w.npy"),
				get_weights_accessor(data_path, "/cnn_data/googlenet_model/loss3/loss3_classifier_b.npy"))
		<< SoftmaxLayer()
		<< OutputLayer(get_output_accessor(label, 5));

    }			

    BranchLayer get_inception_node(const std::string &data_path, std::string &&param_path,
                                   unsigned int a_filt,
                                   std::tuple<unsigned int, unsigned int> b_filters,
                                   std::tuple<unsigned int, unsigned int> c_filters,
                                   unsigned int d_filt, Stream &graph)
    {
        std::string total_path = "/cnn_data/googlenet_model/" + param_path + "/" + param_path + "_";
        SubStream   i_a(graph);
        i_a << ConvolutionLayer(
                1U, 1U, a_filt,
                get_weights_accessor(data_path, total_path + "1x1_w.npy"),
                get_weights_accessor(data_path, total_path + "1x1_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b(graph);
        i_b << ConvolutionLayer(
                1U, 1U, std::get<0>(b_filters),
                get_weights_accessor(data_path, total_path + "3x3_reduce_w.npy"),
                get_weights_accessor(data_path, total_path + "3x3_reduce_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(
                3U, 3U, std::get<1>(b_filters),
                get_weights_accessor(data_path, total_path + "3x3_w.npy"),
                get_weights_accessor(data_path, total_path + "3x3_b.npy"),
                PadStrideInfo(1, 1, 1, 1))
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_c(graph);
        i_c << ConvolutionLayer(
                1U, 1U, std::get<0>(c_filters),
                get_weights_accessor(data_path, total_path + "5x5_reduce_w.npy"),
                get_weights_accessor(data_path, total_path + "5x5_reduce_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(
                5U, 5U, std::get<1>(c_filters),
                get_weights_accessor(data_path, total_path + "5x5_w.npy"),
                get_weights_accessor(data_path, total_path + "5x5_b.npy"),
                PadStrideInfo(1, 1, 2, 2))
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_d(graph);
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL)))
            << ConvolutionLayer(
                1U, 1U, d_filt,
                get_weights_accessor(data_path, total_path + "pool_proj_w.npy"),
                get_weights_accessor(data_path, total_path + "pool_proj_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d));
    }
};

/** Main program for Googlenet
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Target (0 = NEON, 1 = OpenCL, 2 = OpenCL with Tuner), [optional] Path to the weights folder, [optional] image, [optional] labels, [optional] Fast math for convolution layer (0 = DISABLED, 1 = ENABLED) )
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphGooglenetExample>(argc, argv);
}
