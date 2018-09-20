#!/bin/bash

scons Werror=1 debug=0 asserts=0 neon=1 opencl=1 examples=1 os=linux arch=arm64-v8a -j8

