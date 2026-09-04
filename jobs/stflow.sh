#!/bin/bash
#SBATCH --job-name=stflow
#SBATCH --qos=qsong1
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
# STFlow (ICML'25 Spotlight) —— 按作者发布的方式运行, 不做任何改写。
# 关键: --feature_encoder resnet50_trunc 无需 gated 权重(UNI/GigaPath 都要申请)。
# source_dataroot 就是每个队列目录, 与我们盘上 bench_data/<COHORT>/ 布局天然一致。
set -u
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh
conda activate hest
V=/blue/qsong1/wang.qing/systema4ST/venv_stmethods
source $V/bin/activate
M=/blue/qsong1/wang.qing/systema4ST/methods/STFlow
B=/blue/qsong1/wang.qing/he2st/HEST/eval/bench_data
OUT=/blue/qsong1/wang.qing/systema4ST/stflow_run
mkdir -p $OUT/embed $OUT/results
export HF_HOME=/blue/qsong1/wang.qing/systema4ST/.hf
echo "节点 $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== 1. 安装 STFlow ==="
cd $M && python -m pip install -q --no-input -e . 2>&1 | tail -3
python -c "import stflow, torch; print('  stflow ok | torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 先用两个最小队列冒烟(SKCM/HCC 各 2 样本), 通过再铺开 —— 避免 10 队列×100 epoch 白跑
SMOKE="SKCM HCC"
echo "=== 2. 特征提取(resnet50_trunc) ==="
cd $M/stflow/app/hest
for C in $SMOKE; do
  echo "---- $C ----"
  python -u benchmark.py --datasets $C \
      --encoders resnet50_trunc \
      --source_dataroot $B/$C \
      --embed_dataroot $OUT/embed \
      --weights_root $OUT/weights \
      --results_dir $OUT/results \
      --exp_code r50_$C --batch_size 128 2>&1 | tail -25
done

echo "=== 3. STFlow 训练(冒烟: 10 epoch) ==="
cd $M/stflow/app/flow
for C in $SMOKE; do
  echo "---- $C ----"
  python -u train.py --datasets $C --feature_encoder resnet50_trunc \
      --source_dataroot $B/$C --embed_dataroot $OUT/embed \
      --save_dir $OUT/results --exp_code smoke_$C \
      --epochs 10 --batch_size 2 --num_workers 4 2>&1 | tail -30
done
echo "######## STFLOW SMOKE DONE ########"
