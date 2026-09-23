#!/bin/bash
#SBATCH -J dsenv2
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 2 --mem=8G -t 1:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
# 作者包不 pip 安装（其 pyproject 的 license classifier 被新 setuptools 拒绝），改用 PYTHONPATH 直接导入仓库；只补 lightning。
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate deepspot
cd /path/to/project
python -c "import lightning; print('lightning', lightning.__version__)" 2>/dev/null || pip install --no-deps "lightning==$(python -c 'import pytorch_lightning as pl;print(pl.__version__)')" || pip install --no-deps lightning
export PYTHONPATH=/path/to/project/methods/DeepSpot:${PYTHONPATH:-}
python - <<PY
import lightning, torch, inspect
from deepspot.spot.model import DeepSpot
print("deepspot OK  lightning", lightning.__version__, " torch", torch.__version__)
print(inspect.signature(DeepSpot.__init__))
PY
