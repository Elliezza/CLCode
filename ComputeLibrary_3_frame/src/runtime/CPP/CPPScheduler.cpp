/*
 * Copyright (c) 2016-2018 ARM Limited.
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
#include "arm_compute/runtime/CPP/CPPScheduler.h"

#include "arm_compute/core/CPP/ICPPKernel.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/runtime/CPUUtils.h"
#include "arm_compute/core/Rounding.h"

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <system_error>
#include <thread>

namespace arm_compute
{

	size_t get_workload(Window win) { 
		size_t work = 1;
		for(unsigned int dim=0; dim < Coordinates::num_max_dimensions; dim++) {
			work *= win.num_iterations(dim);
		}
		return work;
	}
class Thread
{
public:
    /** Start a new thread. */
    Thread();

    Thread(const Thread &) = delete;
    Thread &operator=(const Thread &) = delete;
    Thread(Thread &&)                 = delete;
    Thread &operator=(Thread &&) = delete;

    /** Destructor. Make the thread join. */
    ~Thread();

    /** Request the worker thread to start executing the given kernel
     * This function will return as soon as the kernel has been sent to the worker thread.
     * wait() needs to be called to ensure the execution is complete.
     */
    void start(ICPPKernel *kernel, const Window &window, const ThreadInfo &info);

    /** Wait for the current kernel execution to complete. */
    void wait();

    /** Function ran by the worker thread. */
    void worker_thread();
    int getId();
    void setId(int id);

    std::thread& get_std_thread() {return _thread;}
private:
    std::thread             _thread;
    ICPPKernel             *_kernel{ nullptr };
    Window                  _window;
    ThreadInfo              _info;
    std::mutex              _m;
    std::condition_variable _cv;
    bool                    _wait_for_work{ false };
    bool                    _job_complete{ true };
    std::exception_ptr      _current_exception;
    int _threadID;
};

int Thread::getId() {
	return _threadID;
}

void Thread::setId(int id) {
	_threadID = id;
}

Thread::Thread()
    : _thread(), _window(), _info(), _m(), _cv(), _current_exception(nullptr),_threadID()
{
    _thread = std::thread(&Thread::worker_thread, this);
}

Thread::~Thread()
{
    // Make sure worker thread has ended
    if(_thread.joinable())
    {
        start(nullptr, Window(), ThreadInfo());
        _thread.join();
    }
}

void Thread::start(ICPPKernel *kernel, const Window &window, const ThreadInfo &info)
{
    _kernel = kernel;
    _window = window;
    _info   = info;

    {
        std::lock_guard<std::mutex> lock(_m);
        _wait_for_work = true;
        _job_complete  = false;
    }
    _cv.notify_one();
}

void Thread::wait()
{
    {
        std::unique_lock<std::mutex> lock(_m);
        _cv.wait(lock, [&] { return _job_complete; });
    }

    if(_current_exception)
    {
        std::rethrow_exception(_current_exception);
    }
}

void Thread::worker_thread()
{
    while(true)
    {
        std::unique_lock<std::mutex> lock(_m);
        _cv.wait(lock, [&] { return _wait_for_work; });
        _wait_for_work = false;

        _current_exception = nullptr;

        // Time to exit
        if(_kernel == nullptr)
        {
            return;
        }

        try
        {
            _window.validate();
            _kernel->run(_window, _info);
        }
        catch(...)
        {
            _current_exception = std::current_exception();
        }

        _job_complete = true;
        lock.unlock();
        _cv.notify_one();
    }
}

CPPScheduler &CPPScheduler::get()
{
    static CPPScheduler scheduler;
    return scheduler;
}

CPPScheduler::CPPScheduler()
    : _num_threads(num_threads_hint()),
      _threads(_num_threads)
{
    unsigned int i, j;
    int rc;

    get_cpu_configuration(_cpu_info);
    set_num_threads(_cpu_info.targetCPU.get_avail_cores());

    if(_cpu_info.targetCPU.targetCPUHint) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset); //clearing cpuset
            auto thread_it =  _threads.begin();

	    for(i=0, j=0; j < _num_threads; i++) {
                    if(_cpu_info.targetCPU.procAvail[i]) {                        
			    CPU_ZERO(&cpuset);
    			    CPU_SET(i, &cpuset);
    			    rc = pthread_setaffinity_np(thread_it->get_std_thread().native_handle(), sizeof(cpu_set_t), &cpuset);
			    thread_it->setId(i);
      			    if(rc != 0) {
				    std::cout << "Error in calling pthread_setaffinity_np " << rc << std::endl;
    			    }
  			    ++thread_it; j++;
                    }
            }
	    
    } 

}

void CPPScheduler::set_num_threads(unsigned int num_threads)
{
    _num_threads = num_threads == 0 ? num_threads_hint() : num_threads;
    _threads.resize(_num_threads);
}

unsigned int CPPScheduler::num_threads() const
{
    return _num_threads;
}

void CPPScheduler::schedule(ICPPKernel *kernel, unsigned int split_dimension)
{
    ARM_COMPUTE_ERROR_ON_MSG(!kernel, "The child class didn't set the kernel");

    /** [Scheduler example] */
    ThreadInfo info;
    info.cpu_info = &_cpu_info;
    bool big_cluster=false;

    const Window      &max_window     = kernel->window();
    const unsigned int num_iterations = max_window.num_iterations(split_dimension);
    info.num_threads                  = std::min(num_iterations, _num_threads);
    info.num_threads = std::min(info.num_threads, 4); //refine threads to 4

  
    if(sched_getcpu()>3) big_cluster=true; 

    if(num_iterations == 0)
    {
        return;
    }

  //  std::cout << kernel->name() << std::endl;

    if(!kernel->is_parallelisable() || info.num_threads == 1)
    {
//	std::cout << "-1, " << get_workload(max_window) << std::endl;
        kernel->run(max_window, info);
    }
    else
    {
        int  t  = 0;
        auto thread_it = _threads.begin();

	if(big_cluster) for(t=0; t < 4; ++thread_it, ++t);
        
	for(t=0; t < info.num_threads; ++t, ++thread_it)
        {

		Window win     = max_window.split_window(split_dimension, t, info.num_threads);
		info.thread_id = t;
		if(get_workload(win) != 0)
			thread_it->start(kernel, win, info);
					
//		std::cout << t << ", " << get_workload(win) << ", no hint, on core: " << thread_it->getId() << std::endl;

	}

        try
        {
		auto thread = _threads.begin();
		if(big_cluster) for(t=0; t < 4; ++thread, ++t);
		
		for(t=0; t < info.num_threads; ++t, ++thread)
		{
			thread->wait();
		}
        }
        catch(const std::system_error &e)
        {
            std::cerr << "Caught system_error with code " << e.code() << " meaning " << e.what() << '\n';
        }
    }
    /** [Scheduler example] */
}
} // namespace arm_compute
