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
using namespace arm_compute::logging;

/** Example demonstrating how to implement Squeezenet's network using the Compute Library's graph API
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Target (0 = NEON, 1 = OpenCL, 2 = OpenCL with Tuner), [optional] Path to the weights folder, [optional] image, [optional] labels, [optional] Fast math for convolution layer (0 = DISABLED, 1 = ENABLED) )
 */
class GraphSqueezenetExample : public Example
{
public:
    void do_setup(int argc, char **argv) override
    {
        std::string data_path; /* Path to the trainable data */
        std::string image;     /* Image data */
        std::string label;     /* Label data */

        // Create a preprocessor object
     //   const std::array<float, 3> mean_rgb{ { 122.68f, 116.67f, 104.01f } };
     //   std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb);

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

	Target    target_hint_gpu = set_target_hint(1);

	initialize_graph(target_hint, data_path, image, label, fast_math_hint, graph0);
	initialize_graph(target_hint, data_path, image, label, fast_math_hint, graph1);
	initialize_graph(target_hint_gpu, data_path, image, label, fast_math_hint, graph2);
/*	initialize_graph(target_hint, data_path, image, label, fast_math_hint, graph3);
	initialize_graph(target_hint, data_path, image, label, fast_math_hint, graph4);
	initialize_graph(target_hint, data_path, image, label, fast_math_hint, graph5);
	initialize_graph(target_hint, data_path, image, label, fast_math_hint, graph6);
	initialize_graph(target_hint, data_path, image, label, fast_math_hint, graph7);*/

        
	// Finalize graph
        GraphConfig config;
        config.use_tuner = (target == 2);
        graph0.finalize(target_hint, config);
	graph1.finalize(target_hint, config);
	graph2.finalize(target_hint_gpu, config);
/*	graph3.finalize(target_hint, config);
	graph4.finalize(target_hint, config);
	graph5.finalize(target_hint, config);
	graph6.finalize(target_hint, config);
	graph7.finalize(target_hint, config);*/

    }


    void do_run() override
    {
	    std::cout << "Start running of the graph" << std::endl;
	    int num_cores = 3;
	    auto tbegin = std::chrono::high_resolution_clock::now();
	    int k = 4;
	    int total = 200;
	    int complete = 0;
	    std::mutex com_m;
	    std::vector<std::thread> workers(num_cores);
	    for(int i = 0; i < num_cores; ++i){
		    workers[i] = std::thread([&, i]{{
				    std::cout << "Creating new threads: " << i << " on CPU:" << sched_getcpu() << std::endl;
				    if (i == 0) {
				    for (int j = 0; j < total; j++) {
				    com_m.lock();complete++;com_m.unlock();
				    if (complete > total){std::cout << "big core: " << j << std::endl;break;}
				    graph0.run();
				    }
				    } else if (i == 1){
				    for (int j = 0; j < total; j++) {
				    com_m.lock();complete++;com_m.unlock();
				    if (complete > total){std::cout << "small core: " << j << std::endl;break;}
				    graph1.run();
				    }
				    } else if (i == 2){
				    for (int j = 0; j < total; j++) {
				    com_m.lock();complete++;com_m.unlock();
				    if (complete > total){ std::cout << "gpu: " << j << std::endl;break;}
				    graph2.run();
				    }
				    }
		    }});
		    cpu_set_t cpuset;
		    CPU_ZERO(&cpuset);
		    CPU_SET((k-i), &cpuset);
		    int rc= pthread_setaffinity_np(workers[i].native_handle(), sizeof(cpu_set_t), &cpuset);
		    if (rc !=0) std::cout << "Error in setting affinity for thread " << i << std::endl;
	    }
	    for(auto&t: workers) t.join();
	    auto tend = std::chrono::high_resolution_clock::now();
	    double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count();
	    double cost = cost0/total;
	    std::cout << "COST:" << cost << std::endl;
    }


private:
    Stream graph0{ 0, "SqueezeNetV1" };
    Stream graph1{ 1, "SqueezeNetV1" };
    Stream graph2{ 2, "SqueezeNetV1" };
    Stream graph3{ 3, "SqueezeNetV1" };
    Stream graph4{ 4, "SqueezeNetV1" };
    Stream graph5{ 5, "SqueezeNetV1" };
    Stream graph6{ 6, "SqueezeNetV1" };
    Stream graph7{ 7, "SqueezeNetV1" };




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
				7U, 7U, 96U,
				get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv1_w.npy"),
				get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv1_b.npy"),
				PadStrideInfo(2, 2, 0, 0))
		<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
		<< PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
		<< ConvolutionLayer(
				1U, 1U, 16U,
				get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire2_squeeze1x1_w.npy"),
				get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire2_squeeze1x1_b.npy"),
				PadStrideInfo(1, 1, 0, 0))
		<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
	graph << get_expand_fire_node(data_path, "fire2", 64U, 64U, graph);
	graph << ConvolutionLayer(
			1U, 1U, 16U,
			get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire3_squeeze1x1_w.npy"),
			get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire3_squeeze1x1_b.npy"),
			PadStrideInfo(1, 1, 0, 0))
		<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
	graph << get_expand_fire_node(data_path, "fire3", 64U, 64U, graph);
	graph << ConvolutionLayer(
			1U, 1U, 32U,
			get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire4_squeeze1x1_w.npy"),
			get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire4_squeeze1x1_b.npy"),
			PadStrideInfo(1, 1, 0, 0))
		<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
	graph << get_expand_fire_node(data_path, "fire4", 128U, 128U, graph);
	graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
		<< ConvolutionLayer(
				1U, 1U, 32U,
				get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire5_squeeze1x1_w.npy"),
				get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire5_squeeze1x1_b.npy"),
				PadStrideInfo(1, 1, 0, 0))
		<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
	graph << get_expand_fire_node(data_path, "fire5", 128U, 128U, graph);
	graph << ConvolutionLayer(
			1U, 1U, 48U,
			get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire6_squeeze1x1_w.npy"),
			get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire6_squeeze1x1_b.npy"),
			PadStrideInfo(1, 1, 0, 0))
		<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
	graph << get_expand_fire_node(data_path, "fire6", 192U, 192U, graph);
	graph << ConvolutionLayer(
			1U, 1U, 48U,
			get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire7_squeeze1x1_w.npy"),
			get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire7_squeeze1x1_b.npy"),
			PadStrideInfo(1, 1, 0, 0))
		<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
	graph << get_expand_fire_node(data_path, "fire7", 192U, 192U, graph);
	graph << ConvolutionLayer(
			1U, 1U, 64U,
			get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire8_squeeze1x1_w.npy"),
			get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire8_squeeze1x1_b.npy"),
			PadStrideInfo(1, 1, 0, 0))
		<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
	graph << get_expand_fire_node(data_path, "fire8", 256U, 256U, graph);
	graph << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
		<< ConvolutionLayer(
				1U, 1U, 64U,
				get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire9_squeeze1x1_w.npy"),
				get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire9_squeeze1x1_b.npy"),
				PadStrideInfo(1, 1, 0, 0))
		<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
	graph << get_expand_fire_node(data_path, "fire9", 256U, 256U, graph);
	graph << ConvolutionLayer(
			1U, 1U, 1000U,
			get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv10_w.npy"),
			get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv10_b.npy"),
			PadStrideInfo(1, 1, 0, 0))
		<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
		<< PoolingLayer(PoolingLayerInfo(PoolingType::AVG))
		<< FlattenLayer()
		<< SoftmaxLayer()
		<< OutputLayer(get_output_accessor(label, 5));

    }



    BranchLayer get_expand_fire_node(const std::string &data_path, std::string &&param_path, unsigned int expand1_filt, unsigned int expand3_filt, Stream &graph)
    {
        std::string total_path = "/cnn_data/squeezenet_v1.0_model/" + param_path + "_";
        SubStream   i_a(graph);
        i_a << ConvolutionLayer(
                1U, 1U, expand1_filt,
                get_weights_accessor(data_path, total_path + "expand1x1_w.npy"),
                get_weights_accessor(data_path, total_path + "expand1x1_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubStream i_b(graph);
        i_b << ConvolutionLayer(
                3U, 3U, expand3_filt,
                get_weights_accessor(data_path, total_path + "expand3x3_w.npy"),
                get_weights_accessor(data_path, total_path + "expand3x3_b.npy"),
                PadStrideInfo(1, 1, 1, 1))
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b));
    }
};

/** Main program for Squeezenet v1.0
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
    return arm_compute::utils::run_example<GraphSqueezenetExample>(argc, argv);
}