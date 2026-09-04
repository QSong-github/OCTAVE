#!/bin/bash
#SBATCH --job-name=stflow2
#SBATCH --qos=qsong1
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
# 放弃 venv_stmethods(--system-site-packages 叠加式, 被 pip 拉进的 numpy1-编译 pandas
# 与 hest 的 numpy2 撞 ABI)。统一用自包含的 venv_np1: numpy 1.26.4 全套自洽。
set -u
V=/blue/qsong1/wang.qing/systema4ST/venv_np1
M=/blue/qsong1/wang.qing/systema4ST/methods/STFlow
B=/blue/qsong1/wang.qing/he2st/HEST/eval/bench_data
OUT=/blue/qsong1/wang.qing/systema4ST/stflow_run
mkdir -p $OUT/embed $OUT/results $OUT/weights
export HF_HOME=/blue/qsong1/wang.qing/systema4ST/.hf
source $V/bin/activate
echo "节点 $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== 0. 环境自洽性检查(这一步不过就别往下跑) ==="
python -m pip install -q --no-input huggingface_hub einops 2>&1 | tail -3
python - <<'PY'
import numpy, pandas, torch, sklearn, h5py
print(f"  numpy {numpy.__version__} | pandas {pandas.__version__} | torch {torch.__version__} "
      f"| cuda {torch.cuda.is_available()}")
import pandas as pd; pd.DataFrame({"a":[1,2]}).sum()   # 触发 _libs, ABI 不合会在此炸
print("  ✅ pandas/numpy ABI 自洽")
PY

echo "=== 1. 安装 STFlow ==="
cd $M && python -m pip install -q --no-input --no-deps -e . 2>&1 | tail -3
python -c "import stflow; print('  ✅ stflow')"

SMOKE="SKCM HCC"
echo "=== 2. 特征提取 (resnet50_trunc, 无需 gated 权重) ==="
cd $M/stflow/app/hest
for C in $SMOKE; do
  echo "---- $C ----"
  python -u benchmark.py --datasets $C --encoders resnet50_trunc \
      --source_dataroot $B/$C --embed_dataroot $OUT/embed \
      --weights_root $OUT/weights --results_dir $OUT/results \
      --exp_code r50_$C --batch_size 128 2>&1 | tr '\r' '\n' | grep -avE "%\|" | tail -20
done

echo "=== 3. STFlow 训练冒烟 (10 epoch) ==="
cd $M/stflow/app/flow
for C in $SMOKE; do
  echo "---- $C ----"
  python -u train.py --datasets $C --feature_encoder resnet50_trunc \
      --source_dataroot $B/$C --embed_dataroot $OUT/embed \
      --save_dir $OUT/results --exp_code smoke_$C \
      --epochs 10 --batch_size 2 --num_workers 4 2>&1 | tr '\r' '\n' | grep -avE "%\|" | tail -25
done
echo "######## STFLOW2 DONE ########"
