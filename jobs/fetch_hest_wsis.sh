#!/bin/bash
#SBATCH -J fetchwsi
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 2 --mem=32G -t 36:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
# HEST 基准 72 张 WSI（MahmoodLab/hest, gated，需先登录 Hugging Face），供 TRIPLEX 的 5×5 邻居特征使用。可断点续传。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project; export HF_HUB_ENABLE_HF_TRANSFER=0
python - <<PY
import os, csv, time
from huggingface_hub import snapshot_download
B="/path/to/he2st/HEST/eval/bench_data"; ids=set()
for c in os.listdir(B):
    d=f"{B}/{c}/splits"
    if os.path.isdir(d):
        for f in os.listdir(d):
            for r in csv.DictReader(open(f"{d}/{f}")): ids.add(r["sample_id"])
ids=sorted(ids); print(len(ids),"张", flush=True)
out="/path/to/project/data/hest_wsis"; os.makedirs(out, exist_ok=True)
for k in range(6):
    try:
        snapshot_download("MahmoodLab/hest", repo_type="dataset", allow_patterns=[f"wsis/{i}.tif" for i in ids], local_dir=out, max_workers=2)
        break
    except Exception as e:
        print("重试", k, type(e).__name__, str(e)[:150], flush=True); time.sleep(120)
have=[i for i in ids if os.path.exists(f"{out}/wsis/{i}.tif")]
print(f"完成 {len(have)}/{len(ids)}；合计 {sum(os.path.getsize(f'{out}/wsis/{i}.tif') for i in have)/1e9:.1f} GB", flush=True)
PY
