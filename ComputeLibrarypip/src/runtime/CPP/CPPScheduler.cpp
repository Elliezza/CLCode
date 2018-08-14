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
    std::cout << "No. of threads? " << num_threads_hint() << std::endl; 

    double _factor = 1; //lets assume big cores are "factor" faster than little
// factor is read from file
    std::ifstream iffactor;
    iffactor.open("/root/.hikey960/factor", std::ios::in);
    if(iffactor.is_open()){
	    std::string line;
	    while(bool(getline(iffactor,line))) {
		    if(line.empty()) continue;
		    _factor = std::stod(line,nullptr);
	    }
    }

    std::cout << "Factor setting: " << _factor << std::endl;    
    if(_cpu_info.targetCPU.targetCPUHint) {
            //target CPU Hint is present. Therefore pin threads to core.

            //resize the number of threads to the available cores
            //set_num_threads(_cpu_info.targetCPU.get_avail_cores());

            cpu_set_t cpuset;
            CPU_ZERO(&cpuset); //clearing cpuset
            auto thread_it =  _threads.begin();

	    std::cout << "Number of threads: " << _num_threads << std::endl;
	    //first assign num_thread - 1
            //i is proc id
            //j is thread id
	    for(i=0, j=0; j < _num_threads; i++) {
                    if(_cpu_info.targetCPU.procAvail[i]) {
                        CPU_ZERO(&cpuset);
			CPU_SET(i, &cpuset);
                        rc = pthread_setaffinity_np(thread_it->get_std_thread().native_handle(),
                                sizeof(cpu_set_t), &cpuset);
			thread_it->setId(i);
                        if(rc != 0) {
                            std::cout << "Error in calling pthread_setaffinity_np " << rc << std::endl;
                        }
                        ++thread_it; j++;
                    }
            }
            //last thread should be assigned to last core
            //Hoping it would be big core
	    /*
            for(; i < _cpu_info.targetCPU.procAvail.size(); i++){
                    if(_cpu_info.targetCPU.procAvail[i]) {
                            CPU_SET(i, &cpuset);
                            //rc = pthread_setaffinity_np(pthread_self().native_handle(), sizeof(cpu_set_t), &cpuset);
                            rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
                            if(rc!=0) {
                                   std::cout << "Error calling pthread_setaffinit_np: " << rc << std::endl;
                            }
                    }
            }*/
	    
    } //else no pinning
    

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

    std::cout << kernel->name() << std::endl;
    std::cout << "-1, " << get_workload(kernel->window()) << std::endl;
    kernel->run(kernel->window(), info);

/*
    //bool conv = true;
    const Window      &max_window     = kernel->window();
    const unsigned int num_iterations = max_window.num_iterations(split_dimension);
    info.num_threads                  = std::min(num_iterations, _num_threads);
    std::string target_kernel ("NECol2ImKernel");
    std::string target_kernel2 ("NEIm2ColKernel");
    std::string target_kernel3 ("NEGEMMAssemblyWrapper");
    std::string target_kernel4 ("NESeparableConvolutionHorKernel");
    std::string target_kernel5 ("NESeparableConvolutionVertKernel");
    std::string target_kernel6 ("NEWeightsReshapeKernel");  
    std::string target_kernel7 ("NEGEMMMatrixMultiplyKernel"); */
   
    /*if(get_workload(max_window) < 32) {
	    info.num_threads = 1;
    } else */
    
   /* if((target_kernel.compare(kernel->name()) != 0) || (get_workload(max_window) < 32000)){ //100352, 25800, 16384) {
	    info.num_threads = std::min(info.num_threads, 4);
   }*/
/*
    if(((target_kernel.compare(kernel->name()) != 0) && (target_kernel2.compare(kernel->name()) != 0) && (target_kernel3.compare(kernel->name()) != 0) && (target_kernel4.compare(kernel->name()) != 0) 
    && (target_kernel5.compare(kernel->name()) != 0) ) && (target_kernel6.compare(kernel->name()) != 0) )//&& (target_kernel7.compare(kernel->name()) != 0) )
    {
            info.num_threads = std::min(info.num_threads, 1);
	   // conv = false;
    }*/

 //   info.num_threads = std::min(info.num_threads, 4); //always 4 threads created

    /*
    if((target_kernel.compare(kernel->name()) == 0) && (get_workload(max_window) < 32000)){ //100352, 25800, 16384) {
            info.num_threads = std::min(info.num_threads, 4);
    }
*/

/*    if(get_workload(max_window) < 4096){ //32000, 100352, 25800, 16384) {
		            info.num_threads = std::min(info.num_threads,4);
	}*/
/*
    if(num_iterations == 0)
    {
        return;
    }

  //  std::cout << kernel->name() << std::endl;

    if(!kernel->is_parallelisable() || info.num_threads == 1)
    {
	std::cout << "-1, " << get_workload(max_window) << std::endl;
        kernel->run(max_window, info);
    }
    else
    {
        int  t  = 0;
        int nBig = 0;
        int nLittle = 0;
	int unused = 0;
        auto thread_it = _threads.begin();
        //if num_threads is less that max possible  _num_threads, 
        // we will use only the threads that are assigned to big cores
        if(_num_threads > (unsigned int)info.num_threads) {
                unused = _num_threads - info.num_threads;
                for(; t < unused; ++thread_it, ++t);
                nBig = _cpu_info.targetCPU.get_avail_bigcores(unused);
		nLittle = _cpu_info.targetCPU.get_avail_littlecores(unused);
	} else {

		nBig = _cpu_info.targetCPU.get_avail_bigcores(0);
		nLittle = _cpu_info.targetCPU.get_avail_littlecores(0);
	}
        if(_cpu_info.targetCPU.targetCPUHint && (nBig + nLittle != info.num_threads)) {
                std::cout << "Error: big + little is not equal num_threads\n"; 
        }
        t =0;

	//if (!conv) {thread_it = _threads.begin();}

        
	for(; t < info.num_threads; ++t, ++thread_it)
        {
		if(_cpu_info.targetCPU.targetCPUHint) {
			Window win;
			//if (conv){
				win     = max_window.split_window(split_dimension, t, nBig, nLittle, true, _factor);
			//} else {
			//	win     = max_window.split_window(split_dimension, t, nLittle, nBig, true, _factor);
			//}
			info.thread_id = t;
			if(get_workload(win) != 0)
				thread_it->start(kernel, win, info);
			std::cout << t << ", " << get_workload(win) << " on core: " << thread_it->getId() << std::endl;
		} else {
			Window win     = max_window.split_window(split_dimension, t, info.num_threads);
			info.thread_id = t;
			if(get_workload(win) != 0)
				thread_it->start(kernel, win, info);
			std::cout << t << ", " << get_workload(win) << ", no hint, on core: " << thread_it->getId() << std::endl;
		}

	//	std::cout<< "thread to core:" << thread_it->getId() << std::endl;
	}
*/
        // Run last part on main thread
	/*
	if(_cpu_info.targetCPU.targetCPUHint) {
		Window win     = max_window.split_window(split_dimension, t, nBig, nLittle, true);
		info.thread_id = t;
		kernel->run(win, info);
		std::cout << t << ", " << get_workload(win) << std::endl;
	} else {
		Window win     = max_window.split_window(split_dimension, t, info.num_threads);
		info.thread_id = t;
		kernel->run(win, info);
		std::cout << t << ", " << get_workload(win) << std::endl;
	}
	*/
/*
        try
        {
            for(auto &thread : _threads)
            {
                thread.wait();
            }
        }
        catch(const std::system_error &e)
        {
            std::cerr << "Caught system_error with code " << e.code() << " meaning " << e.what() << '\n';
        }
    }*/
    /** [Scheduler example] */
}
} // namespace arm_compute
