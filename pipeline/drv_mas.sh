#!/bin/bash
cd /path/to/systema4ST
say(){ echo "[$(date +%H:%M:%S)] $*"; }
python3 fix_hfcfg.py
say "清掉用错归一化的 mascaret 产物"
python3 - <<'PY'
import glob, os, shutil, numpy as np
f = sorted(glob.glob("results/hest_emb/*_mascaret.npz"))
d = np.load(f[0], allow_pickle=True); print("  旧嵌入首块均值 %.4f 标准差 %.4f" % (d["X"].mean(), d["X"].std()))
np.save("/tmp/mas_old.npy", d["X"])
for x in f: os.remove(x)
for p in glob.glob("results/hest_floor*_mascaret") + glob.glob("results/hest_effres_ps_mascaret.json"):
    shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
print("  删除嵌入 %d 个及其全部下游" % len(f))
PY
cat > jobs/masemb.sh <<'SH'
#!/bin/bash
#SBATCH -J masemb
#SBATCH --qos=YOUR_QOS --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1 -c 6 --mem=96G -t 24:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%j.out
set -eu
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
python -u src/hest_embed_v2.py --encoder mascaret --batch 128
SH
say "重抽 mascaret"
sbatch jobs/masemb.sh >/dev/null
while [ "$(squeue -u $USER -r -h -n masemb | wc -l)" -gt 0 ]; do sleep 60; done
grep -h "自带统计量" "$(ls -t logs/masemb_*.out|head -1)" || true
N=$(ls results/hest_emb/*_mascaret.npz 2>/dev/null | wc -l); echo "  嵌入 $N/72"
[ "$N" -ge 72 ] || { tail -15 "$(ls -t logs/masemb_*.out|head -1)"; exit 1; }
python3 - <<'PY'
import glob, numpy as np
d = np.load(sorted(glob.glob("results/hest_emb/*_mascaret.npz"))[0], allow_pickle=True)
o = np.load("/tmp/mas_old.npy")
print("  新嵌入首块均值 %.4f 标准差 %.4f；与旧版最大逐元素差 %.4f" %
      (d["X"].mean(), d["X"].std(), np.abs(d["X"] - o).max()))
assert np.abs(d["X"] - o).max() > 1e-3, "归一化改动没有生效"
PY
say "重跑 mascaret 下游"
cat > jobs/masds.sh <<'SH'
#!/bin/bash
#SBATCH -J masds
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION
#SBATCH --array=0-40 -c 2 --mem=12G -t 8:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%A_%a.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST
export OMP_NUM_THREADS=2
COH=(CCRCC COAD HCC IDC LUNG LYMPH_IDC PAAD PRAD READ SKCM); KS=(10 50 200 800)
I=$SLURM_ARRAY_TASK_ID
[ "$I" = 40 ] && exec python -u src/hest_effres_ps.py --encoder mascaret --out results/hest_effres_ps_mascaret.json
K=${KS[$((I/10))]}; C=${COH[$((I%10))]}
if [ "$K" = "50" ]; then OUT=results/hest_floor_mascaret; else OUT=results/hest_floor_k${K}_mascaret; fi
mkdir -p $OUT
python -u src/hest_floor.py --encoder mascaret --cohort "$C" --k "$K" --out "$OUT/$C.json"
SH
sbatch jobs/masds.sh >/dev/null
while [ "$(squeue -u $USER -r -h | wc -l)" -gt 0 ]; do sleep 60; done
say "全量重算"
sbatch jobs/chain27.sh >/dev/null
while [ "$(squeue -u $USER -r -h -n chain27 | wc -l)" -gt 0 ]; do sleep 20; done
sbatch jobs/agg27.sh >/dev/null
while [ "$(squeue -u $USER -r -h -n agg27 | wc -l)" -gt 0 ]; do sleep 20; done
cat "$(ls -t logs/agg27_*.out | head -1)"
