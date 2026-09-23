#!/bin/bash
#SBATCH -J dsenv
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 2 --mem=16G -t 3:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
# DeepSpot 专用环境：克隆 hest（不动共享环境），补 lightning，--no-deps 装作者包
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
cd /path/to/project
[ -d /path/to/miniconda3/envs/deepspot ] || conda create -y -n deepspot --clone hest
conda activate deepspot
python -c "import lightning" 2>/dev/null || pip install --no-deps "lightning==$(python -c 'import pytorch_lightning as pl;print(pl.__version__)')" || pip install --no-deps lightning
mkdir -p methods && cd methods
[ -d DeepSpot ] || git clone --depth 1 https://github.com/ratschlab/DeepSpot.git
cd DeepSpot && git log -1 --format="%H %cd" && pip install --no-deps -e .
python - <<PY
import lightning, torch
from deepspot.spot.model import DeepSpot
import inspect; print("deepspot OK  lightning", lightning.__version__, " torch", torch.__version__)
print(inspect.signature(DeepSpot.__init__))
PY
