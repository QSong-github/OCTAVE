#!/bin/bash
cd /path/to/project
ENC=$(ls results/hest_effres_ps_*.json | sed 's#.*/hest_effres_ps_##; s#\.json##' | grep -v '^omiclip_raw$' | tr '\n' ' ')
N=$(echo $ENC | wc -w); echo "编码器 $N 个"
cat > jobs/blk.sh <<SH
#!/bin/bash
#SBATCH -J blk
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION --array=0-$((N-1)) -c 4 --mem=24G -t 4:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project; export OMP_NUM_THREADS=4 BLK_K=20,50,200
E=($ENC); N=\${E[\$SLURM_ARRAY_TASK_ID]}
[ -s "results/hest_blocks_\${N}.json" ] && exit 0
python -u hest_blocks.py "\$N"
SH
sbatch jobs/blk.sh >/dev/null
while [ "$(squeue -u $USER -r -h -n blk | wc -l)" -gt 0 ]; do sleep 60; done
echo "产出 $(ls results/hest_blocks_*.json 2>/dev/null | wc -l)/$N  Traceback $(grep -l Traceback logs/blk_*.out 2>/dev/null | wc -l)"
