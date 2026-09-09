#!/bin/bash
cd /path/to/systema4ST
for p in $(pgrep -f "bash jobs/cpu_governor.sh"); do kill $p 2>/dev/null; done; sleep 1
scancel -t PENDING -n xenmultiall -u $USER; sleep 2
echo "取消待运行部分后仍在跑: $(squeue -u $USER -h -t R -n xenmultiall | wc -l)"
idx() { python3 -c "import sys; regs=[int(r) for r in sys.argv[1].split()]; print(','.join(str(e*16+r) for e in range(45) for r in regs))" "$1"; }
A=$(sbatch --parsable --mem=12G --array=$(idx "0 1 2 3 4 5 6 7 8 9 10 11 14 15") jobs/xen_multi_an_tier.sh); echo "tier 12G (14 区域)=$A"
B=$(sbatch --parsable --mem=16G --array=$(idx "13") jobs/xen_multi_an_tier.sh); echo "tier 16G (Ovarian XRrun)=$B"
C=$(sbatch --parsable --mem=26G --array=$(idx "12") jobs/xen_multi_an_tier.sh); echo "tier 26G (Cervical)=$C"
nohup bash jobs/cpu_governor.sh >> logs/cpu_governor.log 2>&1 &
sleep 5; tail -1 logs/cpu_governor.log; squeue -u $USER -h -n xenmultiall -o "%i %t %r" | sed 's/_[0-9]* / /' | awk '{print $2, $3}' | sort | uniq -c | tr "\n" ";"; echo
