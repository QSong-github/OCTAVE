#!/bin/bash
cd /path/to/systema4ST
say(){ echo "[$(date +%H:%M:%S)] $*"; }
python3 add_waiv.py
cat > jobs/waivemb.sh <<'SH'
#!/bin/bash
#SBATCH -J waivemb
#SBATCH --qos=YOUR_QOS --partition=hpg-b200,hpg-rtx6000,hpg-turin
#SBATCH --gres=gpu:1 --array=0-1 -c 6 --mem=96G -t 24:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%A_%a.out
set -eu
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
E=(phaet mascaret); N=${E[$SLURM_ARRAY_TASK_ID]}
echo "节点 $(hostname)  编码器 $N"
python -u src/hest_embed_v2.py --encoder "$N" --batch 128
SH
say "阶段1 抽嵌入"
sbatch jobs/waivemb.sh >/dev/null
while [ "$(squeue -u $USER -r -h -n waivemb | wc -l)" -gt 0 ]; do sleep 60; done
for e in phaet mascaret; do echo "  $e: $(ls results/hest_emb/*_${e}.npz 2>/dev/null | wc -l)/72"; done
OKN=0; for e in phaet mascaret; do [ "$(ls results/hest_emb/*_${e}.npz 2>/dev/null|wc -l)" -ge 72 ] && OKN=$((OKN+1)); done
[ "$OKN" = 2 ] || { say "嵌入未齐"; tail -8 "$(ls -t logs/waivemb_*_0.out|head -1)"; exit 1; }
say "阶段2 下游"
cat > jobs/waivds.sh <<'SH'
#!/bin/bash
#SBATCH -J waivds
#SBATCH --qos=YOUR_QOS --partition=hpg-default
#SBATCH --array=0-81 -c 2 --mem=12G -t 8:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%A_%a.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST
export OMP_NUM_THREADS=2
E=(phaet mascaret); COH=(CCRCC COAD HCC IDC LUNG LYMPH_IDC PAAD PRAD READ SKCM); KS=(10 50 200 800)
I=$SLURM_ARRAY_TASK_ID
if [ "$I" -ge 80 ]; then
  N=${E[$((I-80))]}
  [ -s "results/hest_effres_ps_${N}.json" ] && exit 0
  exec python -u src/hest_effres_ps.py --encoder "$N" --out "results/hest_effres_ps_${N}.json"
fi
N=${E[$((I/40))]}; J=$((I%40)); K=${KS[$((J/10))]}; C=${COH[$((J%10))]}
if [ "$K" = "50" ]; then OUT=results/hest_floor_${N}; else OUT=results/hest_floor_k${K}_${N}; fi
mkdir -p $OUT; [ -s "$OUT/$C.json" ] && exit 0
python -u src/hest_floor.py --encoder "$N" --cohort "$C" --k "$K" --out "$OUT/$C.json"
SH
sbatch jobs/waivds.sh >/dev/null
while [ "$(squeue -u $USER -r -h | wc -l)" -gt 0 ]; do sleep 60; done
say "阶段3 全量重算"
sbatch jobs/chain27.sh >/dev/null
while [ "$(squeue -u $USER -r -h -n chain27 | wc -l)" -gt 0 ]; do sleep 20; done
sbatch jobs/agg27.sh >/dev/null
while [ "$(squeue -u $USER -r -h -n agg27 | wc -l)" -gt 0 ]; do sleep 20; done
cat "$(ls -t logs/agg27_*.out | head -1)"
