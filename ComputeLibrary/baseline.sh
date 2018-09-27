#!/bin/bash
rm res.log

taskset -c 4-7 ./build/examples/graph_alexnet | tee -a res.log
sleep 10

taskset -c 4-7 ./build/examples/graph_googlenet | tee -a res.log
sleep 10

taskset -c 4-7 ./build/examples/graph_mobilenet | tee -a res.log
sleep 10

taskset -c 4-7 ./build/examples/graph_mobilenet_qasymm8 | tee -a res.log
sleep 10

taskset -c 4-7 ./build/examples/graph_resnet50 | tee -a res.log
sleep 10

taskset -c 4-7 ./build/examples/graph_squeezenet | tee -a res.log
sleep 10


taskset -c 0-3 ./build/examples/graph_alexnet | tee -a res.log
sleep 10

taskset -c 0-3 ./build/examples/graph_googlenet | tee -a res.log
sleep 10

taskset -c 0-3 ./build/examples/graph_mobilenet | tee -a res.log
sleep 10

taskset -c 0-3 ./build/examples/graph_mobilenet_qasymm8 | tee -a res.log
sleep 10

taskset -c 0-3 ./build/examples/graph_resnet50 | tee -a res.log
sleep 10

taskset -c 0-3 ./build/examples/graph_squeezenet | tee -a res.log
sleep 10


taskset -c 4-7 ./build/examples/graph_alexnet 1 | tee -a res.log
sleep 10

taskset -c 4-7 ./build/examples/graph_googlenet 1 | tee -a res.log
sleep 10

taskset -c 4-7 ./build/examples/graph_mobilenet 1 | tee -a res.log
sleep 10

taskset -c 4-7 ./build/examples/graph_mobilenet_qasymm8 1 | tee -a res.log
sleep 10

taskset -c 4-7 ./build/examples/graph_resnet50 1 | tee -a res.log
sleep 10

taskset -c 4-7 ./build/examples/graph_squeezenet 1 | tee -a res.log
sleep 10

grep "COST" res.log

