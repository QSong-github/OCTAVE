#!/bin/bash
#SBATCH --job-name=mset
#SBATCH --qos=qsong1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
# 方法集可行性勘察: 计算节点上先测网络, 再逐个 clone 并读依赖。
# 不在这里建环境(每个 env 单独作业), 只回答"哪些能装、装了要什么"。
set -u
D=/blue/qsong1/wang.qing/systema4ST/methods
mkdir -p $D; cd $D
echo "节点 $(hostname)  作业 $SLURM_JOB_ID"
echo "=== 网络连通性 ==="
curl -sS -m 20 -o /dev/null -w "github %{http_code}\n" https://github.com || echo "github 不可达"
curl -sS -m 20 -o /dev/null -w "hf     %{http_code}\n" https://huggingface.co || echo "hf 不可达"

declare -A REPO=(
  [BLEEP]="https://github.com/bowang-lab/BLEEP"
  [STNet]="https://github.com/bryanhe/ST-Net"
  [HisToGene]="https://github.com/maxpmx/HisToGene"
  [Hist2ST]="https://github.com/biomed-AI/Hist2ST"
  [DeepSpot]="https://github.com/ratschlab/he2st"
  [STFlow]="https://github.com/Graph-and-Geometric-Learning/STFlow"
  [THItoGene]="https://github.com/yrjia1015/THItoGene"
)
for k in "${!REPO[@]}"; do
  echo "######## $k ########"
  if [ -d "$k/.git" ]; then echo "  已存在, 跳过 clone"; else
    timeout 300 git clone --depth 1 -q "${REPO[$k]}" "$k" 2>&1 | head -3 || echo "  ❌ clone 失败"
  fi
  if [ -d "$k" ]; then
    echo "  顶层: $(ls $k 2>/dev/null | head -8 | tr '\n' ' ')"
    for f in requirements.txt environment.yml environment.yaml setup.py pyproject.toml; do
      [ -f "$k/$f" ] && echo "  ✓ $f" && head -25 "$k/$f" | sed 's/^/      /'
    done
    ck=$(find $k -maxdepth 2 -iname "*.ckpt" -o -maxdepth 2 -iname "*.pth" -o -maxdepth 2 -iname "*.pt" 2>/dev/null | head -3)
    [ -n "$ck" ] && echo "  权重: $ck"
  fi
done
echo "######## SURVEY DONE ########"
