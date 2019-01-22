cd /sys/devices/system/cpu

echo $1 > cpu0/online
echo $1 > cpu1/online
echo $1 > cpu2/online
echo $1 > cpu3/online

cat online


