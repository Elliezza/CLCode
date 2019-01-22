cd /sys/devices/system/cpu/cpu0/cpufreq
chmod 777 scaling_max_freq scaling_min_freq scaling_cur_freq scaling_governor
echo ondemand > scaling_governor 
echo 1844000 > scaling_max_freq
echo 509000 > scaling_min_freq
cat cpuinfo_cur_freq

cd /sys/devices/system/cpu/cpu4/cpufreq
chmod 777 scaling_max_freq scaling_min_freq scaling_cur_freq scaling_governor
echo ondemand > scaling_governor 
echo 2362000 > scaling_max_freq
echo 682000 > scaling_min_freq
cat cpuinfo_cur_freq
