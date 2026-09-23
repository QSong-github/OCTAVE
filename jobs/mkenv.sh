#!/bin/bash
#SBATCH --job-name=mkenv
#SBATCH --qos=YOUR_QOS
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/project/logs/%x_%j.out
# 建一个共享的 stmethods 环境, 再逐个 import 各方法的入口模块。
# 不照搬 DeepSpot 那份 pin 死的 conda export(400+ 行含 awscli, 必然解不动)。
# 能在共享环境跑通的就共用; 跑不通的下一轮单独建 env。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh
ENV=/path/to/miniconda3/envs/stmethods
M=/path/to/project/methods
echo "节点 $(hostname)"
if [ ! -d "$ENV" ]; then
  conda create -y -p $ENV python=3.10 || exit 1
  conda activate $ENV
  pip install -q --no-input torch torchvision --index-url https://download.pytorch.org/whl/cu121 || echo "torch 装失败"
  pip install -q --no-input numpy scipy pandas scikit-learn h5py anndata scanpy \
      timm einops opencv-python-headless pillow matplotlib tqdm transformers \
      torch-geometric pytorch-lightning wandb || echo "部分依赖失败"
else
  conda activate $ENV
fi
echo "=== 环境就绪 ==="
python -c "import torch,scanpy,timm,einops; print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

echo "=== 逐方法 import 冒烟 ==="
cd $M
for k in STFlow STNet HisToGene Hist2ST THItoGene BLEEP DeepSpot; do
  echo "######## $k ########"
  python - <<PY 2>&1 | head -6
import sys, os, glob, importlib.util
d = "$M/$k"
sys.path.insert(0, d)
cands = [p for p in glob.glob(os.path.join(d, "*.py"))
         if os.path.basename(p) not in ("setup.py",)]
ok, bad = [], []
for p in sorted(cands)[:8]:
    n = os.path.basename(p)[:-3]
    try:
        spec = importlib.util.spec_from_file_location(n, p)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        ok.append(n)
    except Exception as e:
        bad.append(f"{n}: {type(e).__name__}: {str(e)[:70]}")
print("  ✅ 可导入:", ok if ok else "(无顶层 .py)")
for b in bad[:4]: print("  ❌", b)
PY
done
echo "######## MKENV DONE ########"
