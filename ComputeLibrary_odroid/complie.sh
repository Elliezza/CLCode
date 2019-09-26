#!/bin/bash

scons Werror=0 debug=0 asserts=0 neon=1 opencl=1 examples=1 os=linux arch=armv7a -j8

