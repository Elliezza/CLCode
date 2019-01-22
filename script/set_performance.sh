cd /sys/devices/system/cpu/cpu0/cpufreq
chmod 777 scaling_max_freq scaling_min_freq scaling_cur_freq scaling_governor
echo performance > scaling_governor 
echo 1844000 > scaling_max_freq
echo 1844000 > scaling_min_freq
cat cpuinfo_cur_freq

cd /sys/devices/system/cpu/cpu4/cpufreq
chmod 777 scaling_max_freq scaling_min_freq scaling_cur_freq scaling_governor
echo performance > scaling_governor 
echo 2362000 > scaling_max_freq
echo 2362000 > scaling_min_freq
cat cpuinfo_cur_freq

cd /sys/devices/platform/e82c0000.mali/devfreq/e82c0000.mali
echo 767000000 > min_freq
echo 767000000 > max_freq
cat cur_freq
