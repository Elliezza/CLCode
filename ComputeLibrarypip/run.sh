#!/bin/bash
echo 0xFF > /root/.hikey960/cpu_hint
echo $1 > /root/.hikey960/split

./build/examples/$2
