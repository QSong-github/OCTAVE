#!/bin/bash
cd /path/to/project
say(){ echo "[$(date +%H:%M:%S)] $*"; }
python3 add_musk.py
cat > jobs/muskemb.sh <<'SH'
#!/bin/bash
#SBATCH -J muskemb
#SBATCH --qos=YOUR_QOS --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1 -c 6 --mem=96G -t 36:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
set -eu
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
python -u src/hest_embed_v2.py --encoder musk --batch 48
SH
say "阶段1 抽 MUSK 嵌入"
sbatch jobs/muskemb.sh >/dev/null
while [ "$(squeue -u $USER -r -h -n muskemb | wc -l)" -gt 0 ]; do sleep 60; done
N=$(ls results/hest_emb/*_musk.npz 2>/dev/null | wc -l); say "  嵌入 $N/72"
[ "$N" -ge 72 ] || { say "未齐，中止"; tail -6 "$(ls -t logs/muskemb_*.out|head -1)"; exit 1; }
say "阶段2 下游"
cat > jobs/muskps.sh <<'SH'
#!/bin/bash
#SBATCH -J muskps
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION
#SBATCH -c 4 --mem=48G -t 8:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
set -eu
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
python -u src/hest_effres_ps.py --encoder musk --out results/hest_effres_ps_musk.json
SH
cat > jobs/muskfloor.sh <<'SH'
#!/bin/bash
#SBATCH -J muskfloor
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION
#SBATCH --array=0-39 -c 2 --mem=12G -t 8:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project
COH=(CCRCC COAD HCC IDC LUNG LYMPH_IDC PAAD PRAD READ SKCM); KS=(10 50 200 800)
I=$SLURM_ARRAY_TASK_ID; K=${KS[$((I/10))]}; C=${COH[$((I%10))]}
if [ "$K" = "50" ]; then OUT=results/hest_floor_musk; else OUT=results/hest_floor_k${K}_musk; fi
mkdir -p $OUT; [ -s "$OUT/$C.json" ] && exit 0
python -u src/hest_floor.py --encoder musk --cohort "$C" --k "$K" --out "$OUT/$C.json"
SH
sbatch jobs/muskps.sh >/dev/null; sbatch jobs/muskfloor.sh >/dev/null
while [ "$(squeue -u $USER -r -h | wc -l)" -gt 0 ]; do sleep 60; done
say "阶段3 全量重算"
sbatch jobs/chain27.sh >/dev/null
while [ "$(squeue -u $USER -r -h -n chain27 | wc -l)" -gt 0 ]; do sleep 20; done
sbatch jobs/agg27.sh >/dev/null
while [ "$(squeue -u $USER -r -h -n agg27 | wc -l)" -gt 0 ]; do sleep 20; done
cat "$(ls -t logs/agg27_*.out | head -1)"
