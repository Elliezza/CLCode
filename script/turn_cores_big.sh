cd /sys/devices/system/cpu

echo $1 > cpu4/online
echo $1 > cpu5/online
echo $1 > cpu6/online
echo $1 > cpu7/online

cat online


