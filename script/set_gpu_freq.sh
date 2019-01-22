cd /sys/devices/platform/e82c0000.mali/devfreq/e82c0000.mali
echo 767000000 > min_freq
echo 767000000 > max_freq
cat cur_freq
