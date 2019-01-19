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
#include <thread>
#include <cstdlib>

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

/** Example demonstrating how to implement ResNet50 network using the Compute Library's graph API
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Target (0 = NEON, 1 = OpenCL, 2 = OpenCL with Tuner), [optional] Path to the weights folder, [optional] image, [optional] labels, [optional] Fast math for convolution layer (0 = DISABLED, 1 = ENABLED) )
 */
class GraphResNet50Example : public Example
{
public:
    void do_setup(int argc, char **argv) override
    {
        std::string data_path; /* Path to the trainable data */
        std::string image;     /* Image data */
        std::string label;     /* Label data */

//        // Create a preprocessor object
//        const std::array<float, 3> mean_rgb{ { 122.68f, 116.67f, 104.01f } };
//        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb,
//                                                                                                                   false /* Do not convert to BGR */);

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
/*	initialize_graph(target_hint, data_path, image, label, fast_math_hint, graph3);
	initialize_graph(target_hint, data_path, image, label, fast_math_hint, graph4);
	initialize_graph(target_hint, data_path, image, label, fast_math_hint, graph5);
	initialize_graph(target_hint, data_path, image, label, fast_math_hint, graph6);
	initialize_graph(target_hint, data_path, image, label, fast_math_hint, graph7);
*/
	// Finalize graph
        GraphConfig config;
        config.use_tuner = (target == 2);
	graph0.finalize(target_hint, config);	
        graph1.finalize(target_hint, config);
	graph2.finalize(target_hint, config);
/*	graph3.finalize(target_hint, config);
	graph4.finalize(target_hint, config);
	graph5.finalize(target_hint, config);
	graph6.finalize(target_hint, config);
	graph7.finalize(target_hint, config);*/
    }

    void do_run() override
    {
        // Creating 4 threads to run on 4 big cores
	std::cout << "Start running of the graph" << std::endl;
	int num_cores = 3;
	auto tbegin = std::chrono::high_resolution_clock::now();
       
	int k = 4; //starting from big cores
       std::vector<std::thread> workers(num_cores);
       for(int i = 0; i < num_cores; ++i){
	       workers[i] = std::thread([&, i]{
		{
			std::cout << "Creating new threads: " << i << " on CPU:" << sched_getcpu() << std::endl;
			if (i == 0) {
				for (int j = 0; j < 10; j++) graph0.run();
			} else if (i == 1){  
				for (int j = 0; j < 10; j++) graph1.run();
			} else if (i == 2){  
				for (int j = 0; j < 10; j++) graph2.run();
			} /*else if (i == 3){
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
	       CPU_SET((k+i), &cpuset);
	       int rc= pthread_setaffinity_np(workers[i].native_handle(), sizeof(cpu_set_t), &cpuset);
	       if (rc !=0) std::cout << "Error in setting affinity for thread " << i << std::endl;
       }

       for(auto&t: workers) t.join();	

       auto tend = std::chrono::high_resolution_clock::now();
       double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count();
       double cost = cost0/30;
       std::cout << "COST:" << cost << std::endl;
    }


private:
    Stream graph0{ 0, "ResNet50" };
    Stream graph1{ 1, "ResNet50" };
    Stream graph2{ 2, "ResNet50" };
/*    Stream graph3{ 3, "ResNet50" };
    Stream graph4{ 4, "ResNet50" };
    Stream graph5{ 5, "ResNet50" };
    Stream graph6{ 6, "ResNet50" };
    Stream graph7{ 7, "ResNet50" }; 

*/
    void initialize_graph(Target &target_hint, const std::string &data_path, const std::string &image, const std::string &label, FastMathHint &fast_math_hint, Stream &graph)
    {
	    // Create a preprocessor object
         const std::array<float, 3> mean_rgb{ { 122.68f, 116.67f, 104.01f } };
	 std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb, false /* Do not convert to BGR */);

    graph << target_hint
	    << fast_math_hint
	    << InputLayer(TensorDescriptor(TensorShape(224U, 224U, 3U, 1U), DataType::F32),
			    get_input_accessor(image, std::move(preprocessor), false /* Do not convert to BGR */))
	    << ConvolutionLayer(
			    7U, 7U, 64U,
			    get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_weights.npy"),
			    std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
			    PadStrideInfo(2, 2, 3, 3))
	    .set_name("conv1/convolution")
	    << BatchNormalizationLayer(
			    get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_BatchNorm_moving_mean.npy"),
			    get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_BatchNorm_moving_variance.npy"),
			    get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_BatchNorm_gamma.npy"),
			    get_weights_accessor(data_path, "/cnn_data/resnet50_model/conv1_BatchNorm_beta.npy"),
			    0.0000100099996416f)
	    .set_name("conv1/BatchNorm")
	    << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name("conv1/Relu")
	    << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR))).set_name("pool1/MaxPool");

    add_residual_block(data_path, "block1", 64, 3, 2, graph);
    add_residual_block(data_path, "block2", 128, 4, 2, graph);
    add_residual_block(data_path, "block3", 256, 6, 2, graph);
    add_residual_block(data_path, "block4", 512, 3, 1, graph);

    graph << PoolingLayer(PoolingLayerInfo(PoolingType::AVG)).set_name("pool5")
	    << ConvolutionLayer(
			    1U, 1U, 1000U,
			    get_weights_accessor(data_path, "/cnn_data/resnet50_model/logits_weights.npy"),
			    get_weights_accessor(data_path, "/cnn_data/resnet50_model/logits_biases.npy"),
			    PadStrideInfo(1, 1, 0, 0))
	    .set_name("logits/convolution")
	    << FlattenLayer().set_name("predictions/Reshape")
	    << SoftmaxLayer().set_name("predictions/Softmax")
	    << OutputLayer(get_output_accessor(label, 5));
    }

    void add_residual_block(const std::string &data_path, const std::string &name, unsigned int base_depth, unsigned int num_units, unsigned int stride, Stream &graph)
    {
        for(unsigned int i = 0; i < num_units; ++i)
        {
            std::stringstream unit_path_ss;
            unit_path_ss << "/cnn_data/resnet50_model/" << name << "_unit_" << (i + 1) << "_bottleneck_v1_";
            std::stringstream unit_name_ss;
            unit_name_ss << name << "/unit" << (i + 1) << "/bottleneck_v1/";

            std::string unit_path = unit_path_ss.str();
            std::string unit_name = unit_name_ss.str();

            unsigned int middle_stride = 1;

            if(i == (num_units - 1))
            {
                middle_stride = stride;
            }

            SubStream right(graph);
	    right << ConvolutionLayer(
                      1U, 1U, base_depth,
                      get_weights_accessor(data_path, unit_path + "conv1_weights.npy"),
                      std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                      PadStrideInfo(1, 1, 0, 0))
                  .set_name(unit_name + "conv1/convolution")
                  << BatchNormalizationLayer(
                      get_weights_accessor(data_path, unit_path + "conv1_BatchNorm_moving_mean.npy"),
                      get_weights_accessor(data_path, unit_path + "conv1_BatchNorm_moving_variance.npy"),
                      get_weights_accessor(data_path, unit_path + "conv1_BatchNorm_gamma.npy"),
                      get_weights_accessor(data_path, unit_path + "conv1_BatchNorm_beta.npy"),
                      0.0000100099996416f)
                  .set_name(unit_name + "conv1/BatchNorm")
                  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "conv1/Relu")

                  << ConvolutionLayer(
                      3U, 3U, base_depth,
                      get_weights_accessor(data_path, unit_path + "conv2_weights.npy"),
                      std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                      PadStrideInfo(middle_stride, middle_stride, 1, 1))
                  .set_name(unit_name + "conv2/convolution")
                  << BatchNormalizationLayer(
                      get_weights_accessor(data_path, unit_path + "conv2_BatchNorm_moving_mean.npy"),
                      get_weights_accessor(data_path, unit_path + "conv2_BatchNorm_moving_variance.npy"),
                      get_weights_accessor(data_path, unit_path + "conv2_BatchNorm_gamma.npy"),
                      get_weights_accessor(data_path, unit_path + "conv2_BatchNorm_beta.npy"),
                      0.0000100099996416f)
                  .set_name(unit_name + "conv2/BatchNorm")
                  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "conv1/Relu")

                  << ConvolutionLayer(
                      1U, 1U, base_depth * 4,
                      get_weights_accessor(data_path, unit_path + "conv3_weights.npy"),
                      std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                      PadStrideInfo(1, 1, 0, 0))
                  .set_name(unit_name + "conv3/convolution")
                  << BatchNormalizationLayer(
                      get_weights_accessor(data_path, unit_path + "conv3_BatchNorm_moving_mean.npy"),
                      get_weights_accessor(data_path, unit_path + "conv3_BatchNorm_moving_variance.npy"),
                      get_weights_accessor(data_path, unit_path + "conv3_BatchNorm_gamma.npy"),
                      get_weights_accessor(data_path, unit_path + "conv3_BatchNorm_beta.npy"),
                      0.0000100099996416f)
                  .set_name(unit_name + "conv2/BatchNorm");
            if(i == 0)
            {
                SubStream left(graph);

                left << ConvolutionLayer(
                         1U, 1U, base_depth * 4,
                         get_weights_accessor(data_path, unit_path + "shortcut_weights.npy"),
                         std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                         PadStrideInfo(1, 1, 0, 0))
                     .set_name(unit_name + "shortcut/convolution")
                     << BatchNormalizationLayer(
                         get_weights_accessor(data_path, unit_path + "shortcut_BatchNorm_moving_mean.npy"),
                         get_weights_accessor(data_path, unit_path + "shortcut_BatchNorm_moving_variance.npy"),
                         get_weights_accessor(data_path, unit_path + "shortcut_BatchNorm_gamma.npy"),
                         get_weights_accessor(data_path, unit_path + "shortcut_BatchNorm_beta.npy"),
                         0.0000100099996416f)
                     .set_name(unit_name + "shortcut/BatchNorm");

                graph << BranchLayer(BranchMergeMethod::ADD, std::move(left), std::move(right)).set_name(unit_name + "add");
            }
            else if(middle_stride > 1)
            {
                SubStream left(graph);

                left << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 1, PadStrideInfo(middle_stride, middle_stride, 0, 0), true)).set_name(unit_name + "shortcut/MaxPool");


                graph << BranchLayer(BranchMergeMethod::ADD, std::move(left), std::move(right)).set_name(unit_name + "add");

            }
            else
            {
                SubStream left(graph);	

                graph << BranchLayer(BranchMergeMethod::ADD, std::move(left), std::move(right)).set_name(unit_name + "add");
            }

            graph << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)).set_name(unit_name + "Relu");

        }
    }
};


/** Main program for ResNet50
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Target (0 = NEON, 1 = OpenCL, 2 = OpenCL with Tuner), [optional] Path to the weights folder, [optional] image, [optional] labels, [optional] Fast math for convolution layer (0 = DISABLED, 1 = ENABLED) )
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphResNet50Example>(argc, argv);
}