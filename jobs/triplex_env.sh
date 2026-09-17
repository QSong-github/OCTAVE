#!/bin/bash
#SBATCH -J tpxenv
#SBATCH --qos=YOUR_QOS --partition=hpg-default -c 2 --mem=16G -t 3:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%j.out
# TRIPLEX 环境：克隆 hest（不动共享环境），--no-deps 补 addict/einops/wandb；flash_attn 用垫片（默认配置不调用）。
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
cd /path/to/systema4ST
[ -d /path/to/miniconda3/envs/triplex ] || conda create -y -n triplex --clone hest
conda activate triplex
for p in addict einops wandb; do python -c "import $p" 2>/dev/null || pip install --no-deps $p; done
python -c "import torchmetrics" 2>/dev/null || pip install --no-deps torchmetrics
export PYTHONPATH=/path/to/systema4ST/methods/triplex_shim:/path/to/systema4ST/methods/TRIPLEX/src:${PYTHONPATH:-}
cd methods/TRIPLEX
python - <<PY
import torch, lightning, torchmetrics, einops, addict
import importlib; m=importlib.import_module("model.TRIPLEX.TRIPLEX"); print("TRIPLEX 模型模块导入 OK; torch", torch.__version__, "lightning", lightning.__version__)
PY
