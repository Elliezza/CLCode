#!/bin/bash
#export NDK=/home/<user-name>/Android/Sdk/ndk-bundle 
#$NDK/build/tools/make_standalone_toolchain.py --arch arm64 --api 23 --stl gnustl --install-dir /ml/cl/toolchains/aarch64

export PATH=/home/siqi/Documents/Research2018/siqiworkspace/toolchains/aarch64/bin:$PATH

CXX=clang++ CC=clang scons Werror=0 debug=0 asserts=0 neon=1 opencl=1 os=android arch=arm64-v8a -j4
