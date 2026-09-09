#!/bin/bash
#SBATCH --job-name=pyginst
#SBATCH --qos=YOUR_QOS
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%j.out
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
echo "== 计算节点是否有外网 =="
curl -sI --max-time 15 https://pypi.org/simple/ -o /dev/null -w "pypi: %{http_code}\n" || echo "pypi 不可达"
pip install --no-cache-dir torch_geometric 2>&1 | tail -6
python - <<'PY'
import torch, torch_geometric
from torch_geometric.nn import HypergraphConv
print("torch", torch.__version__, " torch_geometric", torch_geometric.__version__)
import torch as T
c = HypergraphConv(8, 4)
x = T.randn(6, 8); ei = T.tensor([[0,1,2,3,4,5],[0,0,1,1,2,2]])
print("HypergraphConv 前向输出", tuple(c(x, ei).shape))
PY
