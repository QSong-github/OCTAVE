#!/bin/bash
cd /blue/qsong1/wang.qing/systema4ST
for p in $(pgrep -f "bash jobs/cpu_governor.sh"); do kill $p 2>/dev/null; done; sleep 1
echo "旧调速器剩余进程: $(pgrep -f 'bash jobs/cpu_governor.sh' | wc -l)"
scancel -t PENDING -n xenmultiall -u wang.qing; sleep 2
sed -e 's/ -c 8 --mem=48G/ -c 2 --mem=32G --nice=50000/' -e 's/OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8/OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2/' jobs/xen_multi_an_all.sh > jobs/xen_multi_an_all2.sh
grep -n "SBATCH --qos\|OMP_NUM" jobs/xen_multi_an_all2.sh
J=$(sbatch --parsable jobs/xen_multi_an_all2.sh); echo "新阵列 xenmultiall=$J"
nohup bash jobs/cpu_governor.sh >> logs/cpu_governor.log 2>&1 &
sleep 3; tail -2 logs/cpu_governor.log; squeue -u wang.qing -h -n xenmultiall -o "%i %t %C" | sed 's/_[0-9]* / /' | awk '{print $2, $3}' | sort | uniq -c | tr "\n" ";"; echo
