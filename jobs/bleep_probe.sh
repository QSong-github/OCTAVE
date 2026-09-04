#!/bin/bash
#SBATCH --job-name=bleepprobe
#SBATCH --qos=qsong1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
# 先读死代码里的硬编码假设, 再写适配 —— 这是 STFlow 那八轮换来的教训。
set -u
M=/blue/qsong1/wang.qing/systema4ST/methods/BLEEP
echo "=== config.py(全部硬编码) ==="; cat $M/config.py
echo; echo "=== dataset.py 的数据接口 ==="; grep -nE "def |h5ad|\.npy|\.csv|path|read_|Dataset|__getitem__|image|spatial" $M/dataset.py | head -30
echo; echo "=== BLEEP_main.py 入口与参数 ==="; grep -nE "add_argument|def main|__main__|load|train|build" $M/BLEEP_main.py | head -25
echo; echo "=== 期望的数据目录 ==="; ls $M/GSE240429_data 2>/dev/null | head
