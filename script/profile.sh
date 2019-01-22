#!/bin/bash
 
PREV_TOTAL=( 0 0 0 0 0 0 0 0 0 )
PREV_IDLE=( 0 0 0 0 0 0 0 0 0 )
PREV_USER=( 0 0 0 0 0 0 0 0 0 )
TOTAL_NUM=${#PREV_TOTAL[*]}
PREV_GPU_ACTIVE=0
PREV_GPU_IDLE=0

per="%"
#echo "TOTAL,USER||  CORE0  |  CORE1  |  CORE2  |  CORE3  |  CORE4  |  CORE5  |  CORE6  |  CORE7  |  GPU   |SAMP_PERIOD| TEMP"
echo "TOTAL,USER||CORE0|CORE1|CORE2|CORE3|CORE4|CORE5|CORE6|CORE7| GPU |SAMP_PERIOD| TEMP"
while true; 
do
	# Get the total CPU statistics, discarding the 'cpu ' prefix.

	start=`date +%s.%N`
	for (( k=0; k<$(( $TOTAL_NUM )); k++ ))
	do
		let "N=$k-1"
		if [ $k -eq 0 ]
		then
			CPU=(`sed -n 's/^cpu\s//p' /proc/stat`)
		else
			CPU=(`sed -n 's/^cpu'$N'\s//p' /proc/stat`)
		fi

	IDLE=${CPU[3]} # Just the idle CPU time.
	USER=${CPU[0]} # user activity time

	# Calculate the total CPU time.
	TOTAL=0
	for VALUE in "${CPU[@]}"; do
		let "TOTAL=$TOTAL+$VALUE"
	done

	# Calculate the CPU usage since we last checked.
	let "DIFF_IDLE=$IDLE-${PREV_IDLE[k]}"
	let "DIFF_TOTAL=$TOTAL-${PREV_TOTAL[k]}"
	let "DIFF_USAGE=(1000*($DIFF_TOTAL-$DIFF_IDLE)/$DIFF_TOTAL+5)/10"
	let "DIFF_USAGE2=(1000*($USER - ${PREV_USER[k]})/$DIFF_TOTAL+5)/10"
	if [ $k -eq 0 ]
	then
		#echo -en "CPU: $DIFF_USAGE%, $DIFF_USAGE2%; "
		#echo -en "$DIFF_USAGE%, $DIFF_USAGE2% || "
		printf "%3s%s,%3s%s ||" "$DIFF_USAGE" "$per" "$DIFF_USAGE2" "$per"

	else
		#echo -en "CPU[$N]: $DIFF_USAGE%, $DIFF_USAGE2%; "
		#echo -en "$DIFF_USAGE%, $DIFF_USAGE2% | "
		#printf "%3s%s,%3s%s|" "$DIFF_USAGE" "$per" "$DIFF_USAGE2" "$per"
		printf " %3s%s|" "$DIFF_USAGE2" "$per"
	fi
	# Remember the total and idle CPU times for the next check.
	PREV_TOTAL[k]="$TOTAL"
	PREV_IDLE[k]="$IDLE"
	PREV_USER[k]="$USER"

        done


	#get GPU info

	GPU_ACTIVE=(`cat /sys/devices/platform/e82c0000.mali/power/runtime_active_time`)
	GPU_IDLE=(`cat /sys/devices/platform/e82c0000.mali/power/runtime_suspended_time`)

	let "DIFF_GPU_ACTIVE=$GPU_ACTIVE-$PREV_GPU_ACTIVE"
	let "DIFF_GPU_IDLE=$GPU_IDLE-$PREV_GPU_IDLE"
	let "GPU_UTIL=(1000*$DIFF_GPU_ACTIVE/($DIFF_GPU_ACTIVE+$DIFF_GPU_IDLE+5))/10"
	printf " %3s%s|" "$GPU_UTIL" "$per"

	PREV_GPU_ACTIVE="$GPU_ACTIVE"
	PREV_GPU_IDLE="$GPU_IDLE"
	#get time info
        end=`date +%s.%N`
	runtime=$( echo "$end - $start" | bc -l )
	echo -en "$runtime | "

	#get temperature info
	cat /sys/class/thermal/thermal_zone0/temp

	start=`date +%s.%N`
	
	# Wait before checking again.
	sleep 0.1
done
