#!/bin/bash
# 端到端驱动：等嵌入 → 发下游 → 等下游 → 汇总。全程无人值守。
cd /path/to/systema4ST
say(){ echo "[$(date +%H:%M:%S)] $*"; }

say "阶段1 等 genps 与 omiclip_raw 嵌入"
while [ "$(squeue -u $USER -r -h -n genps,fixup | wc -l)" -gt 0 ]; do sleep 60; done
N=$(ls results/hest_emb/*_omiclip_raw.npz 2>/dev/null | wc -l)
say "  omiclip_raw 嵌入 $N/72; genbio 逐样本 PCC: $(ls results/hest_effres_ps_genbio_pathfm.json 2>/dev/null | wc -l)"

if [ "$N" -ge 72 ]; then
  say "阶段2 发 omiclip_raw 的下游"
  cat > jobs/orps.sh <<'SH'
#!/bin/bash
#SBATCH -J orps
#SBATCH --qos=YOUR_QOS --partition=hpg-default
#SBATCH -c 4 --mem=48G -t 8:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%j.out
set -eu
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
python -u src/hest_effres_ps.py --encoder omiclip_raw --out results/hest_effres_ps_omiclip_raw.json
SH
  cat > jobs/orfloor.sh <<'SH'
#!/bin/bash
#SBATCH -J orfloor
#SBATCH --qos=YOUR_QOS --partition=hpg-default
#SBATCH --array=0-39 -c 2 --mem=12G -t 8:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%A_%a.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
COH=(CCRCC COAD HCC IDC LUNG LYMPH_IDC PAAD PRAD READ SKCM); KS=(10 50 200 800)
I=$SLURM_ARRAY_TASK_ID; K=${KS[$((I/10))]}; C=${COH[$((I%10))]}
if [ "$K" = "50" ]; then OUT=results/hest_floor_omiclip_raw; else OUT=results/hest_floor_k${K}_omiclip_raw; fi
mkdir -p $OUT
[ -s "$OUT/$C.json" ] && exit 0
python -u src/hest_floor.py --encoder omiclip_raw --cohort "$C" --k "$K" --out "$OUT/$C.json"
SH
  sbatch jobs/orps.sh >/dev/null; sbatch jobs/orfloor.sh >/dev/null
  say "  已发 1 + 40 个任务"
else
  say "  omiclip_raw 嵌入未齐($N/72)，跳过其下游"
fi

say "阶段3 等全部作业结束"
while [ "$(squeue -u $USER -r -h | wc -l)" -gt 0 ]; do sleep 60; done

say "阶段4 汇总"
sbatch jobs/agg27.sh >/dev/null
while [ "$(squeue -u $USER -r -h -n agg27 | wc -l)" -gt 0 ]; do sleep 20; done
cat "$(ls -t logs/agg27_*.out | head -1)"
