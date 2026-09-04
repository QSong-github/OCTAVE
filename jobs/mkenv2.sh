#!/bin/bash
#SBATCH --job-name=mkenv2
#SBATCH --qos=qsong1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
# 不用 conda create(上一轮 38895016 报内部错误)。改为在 hest 之上叠一层 venv:
#   --system-site-packages 继承 hest 已有的 torch/scanpy/h5py, 只 pip 装缺的,
#   既不污染 hest, 也绕开 conda 求解器。
set -u
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh
conda activate hest
M=/blue/qsong1/wang.qing/systema4ST/methods
V=/blue/qsong1/wang.qing/systema4ST/venv_stmethods
echo "节点 $(hostname)"

echo "=== 第一步: hest 里已有什么 ==="
python - <<'PY'
import importlib
for m in ["torch","torchvision","scanpy","anndata","h5py","timm","einops","cv2","PIL",
          "transformers","torch_geometric","pytorch_lightning","lightning","wandb",
          "sklearn","scipy","pandas","tqdm","matplotlib","tifffile","openslide"]:
    try:
        mod = importlib.import_module(m)
        print(f"  ✅ {m:20s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  ❌ {m:20s} {type(e).__name__}")
PY

[ -d "$V" ] || python -m venv --system-site-packages "$V"
source "$V/bin/activate"
python -m pip install -q --no-input --upgrade pip
echo "=== 第二步: 补装缺失(只装 hest 没有的) ==="
python -m pip install -q --no-input timm einops opencv-python-headless \
    torch-geometric pytorch-lightning wandb tifffile 2>&1 | tail -5

echo "=== 第三步: 逐方法 import 冒烟 ==="
cd $M
for k in STFlow STNet HisToGene Hist2ST THItoGene BLEEP DeepSpot; do
  echo "######## $k ########"
  python - "$M/$k" <<'PY' 2>&1 | head -8
import sys, os, glob, importlib.util
d = sys.argv[1]; sys.path.insert(0, d)
ok, bad = [], []
for p in sorted(glob.glob(os.path.join(d, "*.py")))[:10]:
    n = os.path.basename(p)[:-3]
    if n == "setup": continue
    try:
        spec = importlib.util.spec_from_file_location(n, p)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        ok.append(n)
    except Exception as e:
        bad.append(f"{n} -> {type(e).__name__}: {str(e)[:60]}")
print("  ✅", ",".join(ok) if ok else "(顶层无可导入 .py)")
for b in bad[:5]: print("  ❌", b)
PY
done
echo "######## MKENV2 DONE ########"
