#!/bin/bash
c=1
end=$1

rm temp.log

while [ $c -le $end ]
do
	cat /sys/class/thermal/thermal_zone0/temp >> temp.log
	sleep 0.1
	(( c++ ))
done

