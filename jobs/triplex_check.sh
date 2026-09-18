#!/bin/bash
#SBATCH -J tpxcheck
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 2 --mem=16G -t 0:30:00
#SBATCH -o /path/to/systema4ST/logs/%x_%j.out
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate triplex
cd /path/to/systema4ST
python - <<PY
import importlib, sys
for m in ["hest","hest.bench","hest.bench.cpath_model_zoo.inference_models","hestcore","hestcore.segmentation","trident","loguru","wget","einops","addict","wandb","torchmetrics","pytorch_lightning"]:
    try:
        mod=importlib.import_module(m); print("OK ", m, getattr(mod,"__version__",""), getattr(mod,"__file__",""))
    except Exception as e: print("缺 ", m, type(e).__name__, str(e)[:80])
try:
    from hest import iter_hest; import inspect; from hest.HESTData import HESTData; print("iter_hest OK; dump_patches 签名:", inspect.signature(HESTData.dump_patches))
except Exception as e: print("iter_hest/dump_patches:", type(e).__name__, str(e)[:120])
PY
