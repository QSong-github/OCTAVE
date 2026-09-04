#!/bin/bash
# 第二批：hibou_b + gigapath_flash 抽嵌入（GPU），path_foundation 由 pfemb 作业产出；三者齐后跑同一套下游。
cd /blue/qsong1/wang.qing/systema4ST
say(){ echo "[$(date +%H:%M:%S)] $*"; }
python3 patch_b3.py || exit 1
E2="hibou_b gigapath_flash"
cat > jobs/b3emb.sh <<SH
#!/bin/bash
#SBATCH -J b3emb
#SBATCH --qos=qsong1 --partition=hpg-b200,hpg-rtx6000,hpg-turin --gres=gpu:1 --array=0-1 -c 6 --mem=16G -t 24:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%A_%a.out
set -eu
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST; export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
export HF_TOKEN=\$(cat /blue/qsong1/wang.qing/.cache/huggingface/token); export HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN; unset HF_HUB_OFFLINE
A=($E2); X=\${A[\$SLURM_ARRAY_TASK_ID]}
[ "\$(ls results/hest_emb/*_\${X}.npz 2>/dev/null | wc -l)" -ge 72 ] && exit 0
python -u src/hest_embed_v2.py --encoder "\$X" --batch 128
SH
sbatch jobs/b3emb.sh >/dev/null; say "b3emb 已发"
# 等三者的嵌入齐全（含 TF 作业产出的 path_foundation）
while :; do
  ok=""; for e in $E2 path_foundation; do [ "$(ls results/hest_emb/*_${e}.npz 2>/dev/null | wc -l)" -ge 72 ] && ok="$ok $e"; done
  n=$(echo $ok | wc -w); [ "$n" -ge 3 ] && break
  # 若相关作业都已不在队列而仍不齐，退出让人看
  if [ "$(squeue -u wang.qing -r -h | grep -cE 'b3emb|pfemb|tfenv')" -eq 0 ]; then say "作业已结束但嵌入未齐: $ok"; break; fi
  sleep 120
done
OK=$(echo $ok); NOK=$(echo $OK | wc -w); say "嵌入齐全 $NOK/3: $OK"; [ "$NOK" -gt 0 ] || exit 1
ALPHAS="0.1 1 10 100 1000 10000 100000 1000000 10000000 100000000 1000000000"
cat > jobs/b3ds.sh <<SH
#!/bin/bash
#SBATCH -J b3ds
#SBATCH --qos=qsong1 --partition=hpg-default --array=0-$((NOK*13-1)) -c 4 --mem=24G -t 8:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%A_%a.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST; export OMP_NUM_THREADS=4 BLK_K=20,50,200
A=($OK); AL=($ALPHAS); I=\$SLURM_ARRAY_TASK_ID; X=\${A[\$((I/13))]}; J=\$((I%13))
if [ \$J = 0 ]; then O=results/hest_effres_ps_\${X}.json; [ -s \$O ] && exit 0; exec python -u src/hest_effres_ps.py --encoder \$X --skip_sigma --out \$O; fi
if [ \$J = 12 ]; then [ -s results/hest_blocks_\${X}.json ] && exit 0; exec python -u hest_blocks.py \$X; fi
a=\${AL[\$((J-1))]}; O=results/ridge_alpha/hest_ra_\${X}_a\${a}.json; [ -s \$O ] && exit 0
python -u src/hest_effres_ps.py --encoder \$X --skip_sigma --ridge_alpha \$a --out \$O
SH
sbatch jobs/b3ds.sh >/dev/null
while [ "$(squeue -u wang.qing -r -h -n b3ds | wc -l)" -gt 0 ]; do sleep 60; done
J=$(sbatch --parsable --qos=qsong1 -p hpg-default -c 2 --mem=16G -t 1:00:00 -J b3agg -o logs/b3agg_%j.out --wrap="source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest; cd /blue/qsong1/wang.qing/systema4ST; python3 ridge_loco.py $(echo $OK | tr ' ' ','); python3 blk_agg.py | tail -8")
while [ "$(squeue -j $J -h | wc -l)" -gt 0 ]; do sleep 20; done; cat logs/b3agg_${J}.out
