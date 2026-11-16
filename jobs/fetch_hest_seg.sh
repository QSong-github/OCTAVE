#!/bin/bash
#SBATCH -J fetchseg
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 2 --mem=8G -t 6:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
# HEST 基准 72 张的 st/{id}.h5ad 与 metadata/{id}.json（TRIPLEX 的 prepare_data.py 走 hest.iter_hest 需要 HEST 原始布局）。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
python - <<PY
import os, csv, time
from huggingface_hub import snapshot_download
B="/path/to/he2st/HEST/eval/bench_data"; ids=set()
for c in os.listdir(B):
    d=f"{B}/{c}/splits"
    if os.path.isdir(d):
        for f in os.listdir(d):
            for r in csv.DictReader(open(f"{d}/{f}")): ids.add(r["sample_id"])
ids=sorted(ids); out="/path/to/project/data/hest_wsis"
pats=[f"tissue_seg/{i}*" for i in ids]
for k in range(6):
    try: snapshot_download("MahmoodLab/hest", repo_type="dataset", allow_patterns=pats, local_dir=out, max_workers=4); break
    except Exception as e: print("重试", k, type(e).__name__, str(e)[:150], flush=True); time.sleep(120)
print("tissue_seg 文件:", len(os.listdir(f"{out}/tissue_seg")) if os.path.isdir(f"{out}/tissue_seg") else 0, flush=True)
PY
