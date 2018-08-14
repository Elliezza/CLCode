/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/graph/frontend/Stream.h"

#include "arm_compute/graph/Utils.h"
#include "arm_compute/graph/frontend/ILayer.h"

namespace arm_compute
{
namespace graph
{
namespace frontend
{
Stream::Stream(size_t id, std::string name)
    : _manager(), _ctx(), _g(id, std::move(name)) 
{
}

void Stream::finalize(Target target, const GraphConfig &config)
{
    PassManager pm = create_default_pass_manager(target);
    _ctx.set_config(config);
    _manager.finalize_graph(_g, _ctx, pm, target);
}

void Stream::run() //execution of graph.run()
{
  /*  int num_threads =2;
    std::vector<std::thread> workers(num_threads);
    for(int i = 0; i < num_threads; ++i){
    	workers[i] = std::thread([&, i]{
		std::cout << "Creating new threads: " << i << " on CPU:" << sched_getcpu() << std::endl;
		for (int j = 0; j < 10; j++) _manager.execute_graph(_g);
        });
       cpu_set_t cpuset;
       CPU_ZERO(&cpuset);
       CPU_SET((4+i), &cpuset);
       int rc= pthread_setaffinity_np(workers[i].native_handle(), sizeof(cpu_set_t), &cpuset);
       if (rc !=0) std::cout << "Error in setting affinity for thread " << i << std::endl;
       }

    for(auto&t: workers) t.join();
*/
    _manager.execute_graph(_g);
   // _manager.execute_graph(_g);
    //multithreading with different graph ID.
    //_manager.execute_workload(_g.id());
}

void Stream::add_layer(ILayer &layer)
{
    auto nid   = layer.create_layer(*this);
    _tail_node = nid;
}

const Graph &Stream::graph() const
{
    return _g;
}

Graph &Stream::graph()
{
    return _g;
}
} // namespace frontend
} // namespace graph
} // namespace arm_compute
