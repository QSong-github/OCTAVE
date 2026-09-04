#!/bin/bash
#SBATCH -J new16emb
#SBATCH --qos=qsong1 --partition=hpg-b200,hpg-rtx6000,hpg-turin
#SBATCH --gres=gpu:1 --array=0-15 -c 6 --mem=16G -t 24:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%A_%a.out
set -eu
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 HF_TOKEN=$(cat /blue/qsong1/wang.qing/.cache/huggingface/token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
A=(kaiko_vitb8 lunit_vits16 lunit_r50_swav lunit_r50_bt lunit_r50_moco ctranspath gpfm retccl dinov2_base dinov2_giant dinov3_vitb16 dinov3_vith16 clip_vitl14 pathgen_clip siglip2 biomedclip); X=${A[$SLURM_ARRAY_TASK_ID]}
[ "$(ls results/hest_emb/*_${X}.npz 2>/dev/null | wc -l)" -ge 72 ] && exit 0
echo "节点 $(hostname) 编码器 $X"
python -u src/hest_embed_v2.py --encoder "$X" --batch 128
