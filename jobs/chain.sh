#!/bin/bash
#SBATCH --job-name=hestchain
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%j.out
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=16
export HF_HOME=/path/to/systema4ST/.hf

echo "######## 1. 阶梯延长到 tmax=8192 (σ 上探至 ~3.5mm, 解除等价σ截断) ########"
python -u src/hest_ladder.py --tmax 8192 --reps 3 --norm log1p || exit 1

echo "######## 2. 追加开放权重的强编码器 ########"
for e in phikon_v2 hibou_b; do
  echo "--- $e ---"
  python -u src/hest_embed.py --encoder $e || echo "[skip] $e 不可用(可能 gated)"
done

echo "######## 3. 全部编码器重跑相关性分析 ########"
for e in resnet50 phikon phikon_v2 hibou_b; do
  f=results/hest_reported_pcc_$e.json
  [ "$e" = resnet50 ] && f=results/hest_reported_pcc.json
  if [ -f "$f" ]; then
    echo "================= $e ================="
    python -u src/hest_corr.py $e
  fi
done
echo "######## CHAIN DONE ########"
