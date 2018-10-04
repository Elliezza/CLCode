#!/bin/bash
c=1
end=$1

rm freq.log

while [ $c -le $end ]
do
	cat /sys/devices/platform/e82c0000.mali/devfreq/e82c0000.mali/cur_freq >> freq.log
	cat /sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_cur_freq >> freq.log
	sleep 0.1
	(( c++ ))
done

