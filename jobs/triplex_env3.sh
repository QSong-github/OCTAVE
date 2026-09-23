#!/bin/bash
#SBATCH -J tpxenv3
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 2 --mem=8G -t 1:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
# TRIPLEX 的 train_utils 固定挂 WandbLogger：装完整 wandb，训练时用 WANDB_MODE=offline（不联网）。
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate triplex
pip install wandb 2>&1 | tail -1; WANDB_MODE=offline python -c "import wandb; print('wandb', wandb.__version__)"
