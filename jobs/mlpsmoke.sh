#!/bin/bash
#SBATCH -J mlpsmoke
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 2 --mem=16G -t 2:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%j.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST; export OMP_NUM_THREADS=2
python -u src/hest_effres_ps.py --encoder ciga --skip_sigma --head mlp --seed 0 --out results/mlp_seeds/hest_mlp_ps_ciga_s0.json
python - <<'PY'
import json
m=json.load(open("results/mlp_seeds/hest_mlp_ps_ciga_s0.json")); r=json.load(open("results/hest_effres_ps_ciga.json"))
print("冒烟对照 ciga: mlp 均值 %.4f (n=%d)  ridge 均值 %.4f" % (m["mean_pcc"], len(m["per_sample_pcc"]), r["mean_pcc"]))
assert len(m["per_sample_pcc"])==72
PY
