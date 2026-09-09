#!/bin/bash
#SBATCH --job-name=mkenv4
#SBATCH --qos=YOUR_QOS
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%j.out
# 三个方法(HisToGene/Hist2ST/THItoGene)依赖 scprep, 其编译产物要求 numpy<2,
# 而 hest 是 numpy 2.x —— venv --system-site-packages 继承了 numpy2, 所以叠加装不可能解决。
# 必须独立环境。这里用纯 venv(不继承系统包) + numpy<2 的整套。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
M=/path/to/systema4ST/methods
V1=/path/to/systema4ST/venv_np1        # numpy<2: HisToGene/Hist2ST/THItoGene
V2=/path/to/systema4ST/venv_stmethods  # 已有: BLEEP/STFlow/ST-Net
echo "节点 $(hostname)"

echo "=== A. numpy<2 独立环境 ==="
[ -d "$V1" ] || python -m venv "$V1"          # 注意: 不加 --system-site-packages
source "$V1/bin/activate"
python -m pip install -q --no-input --upgrade pip
python -m pip install -q --no-input "numpy<2" scipy pandas scikit-learn h5py \
    "anndata<0.11" "scanpy<1.11" scprep easydl torch torchvision timm einops \
    opencv-python-headless tqdm matplotlib pytorch-lightning torchmetrics 2>&1 | tail -6
python -c "import numpy,scipy,scprep; print(f'  numpy {numpy.__version__} scipy {scipy.__version__} scprep ok')"
for k in HisToGene Hist2ST THItoGene; do
  echo "######## $k ########"
  python - "$M/$k" <<'PY' 2>&1 | head -6
import sys, os, glob, importlib.util
d=sys.argv[1]; sys.path.insert(0,d); ok,bad=[],[]
for p in sorted(glob.glob(os.path.join(d,"*.py"))):
    n=os.path.basename(p)[:-3]
    if n=="setup": continue
    try:
        s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); ok.append(n)
    except Exception as e: bad.append(f"{n} -> {type(e).__name__}: {str(e)[:55]}")
print("  ✅", ",".join(ok) if ok else "(无)")
for b in bad[:4]: print("  ❌", b)
PY
done
deactivate

echo "=== B. 补 ST-Net 缺的 argcomplete, 并探明 STFlow / DeepSpot 结构 ==="
source "$V2/bin/activate"
python -m pip install -q --no-input argcomplete 2>&1 | tail -2
cd $M/STNet && python -c "import sys;sys.path.insert(0,'.');import stnet;print('  ✅ stnet ->',[x for x in dir(stnet) if not x.startswith('_')][:10])" 2>&1 | head -3
echo "--- STFlow 结构 ---"
find $M/STFlow -name "*.py" | head -15
echo "--- DeepSpot 结构 ---"
find $M/DeepSpot -maxdepth 2 -name "*.py" | head -15
ls $M/DeepSpot
echo "######## MKENV4 DONE ########"
