#!/bin/bash
#SBATCH --job-name=bleep2
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%j.out
# 三处修正(相对首轮):
#  ① 不再 | tail —— 那会缓冲到进程结束, 1.4 小时零输出无法判断死活
#  ② 每折即时落盘 + 断点续跑 —— 48h 墙钟下"跑完才写"等于超时即全丢
#  ③ epochs 40 → 12 —— L4 上 ResNet50 约 250 img/s, CCRCC 单折 6 万 patch×40ep
#     ≈2.7h, 12 折就 32h。12 epoch 时损失已进平台期(冒烟 5 ep 就从 20.9 降到 6.1)。
#     batch 256 → 160 避开日志里出现过的 OOM 警告。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
echo "节点 $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# 小队列优先: 先拿到覆盖面, 再啃 CCRCC/PRAD 这种大的
python -u src/bleep_hest.py --cohorts SKCM,HCC,LUNG,PAAD,COAD,READ,IDC,LYMPH_IDC \
    --epochs 12 --batch_size 160 --out results/bleep_hest.json
python -u src/bleep_hest.py --cohorts PRAD,CCRCC \
    --epochs 12 --batch_size 160 --out results/bleep_hest.json
echo "######## BLEEP2 DONE ########"
