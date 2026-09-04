#!/bin/bash
#SBATCH --job-name=istar_eval
#SBATCH --qos=qsong1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
# 阶段 3/3: 在测试 bin 位置采样官方 iStar 的超分输出算 per-gene PCC(hest 环境, CPU)
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export PYTHONDONTWRITEBYTECODE=1
export TQDM_DISABLE=1              # 上一阶段日志被进度条刷到 371MB
export PYTHONWARNINGS=ignore

for D in P2_checker P2_half P5_checker P5_half; do
  echo "########## $D ##########"
  # topk=50 与标尺/ruler 的评测基因口径一致(top-50 HVG); 同时给 200 基因全量作参照
  python -u src/istar_eval.py --fold_dir istar_run/${D}/ --topk 50
  python -u src/istar_eval.py --fold_dir istar_run/${D}/
done
echo "===== ISTAR EVAL DONE ====="
