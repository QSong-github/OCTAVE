#!/bin/bash
# 单个/多个编码器的全套：GPU 抽嵌入 → 13 任务下游（官方 ridge、11 档 α、块预言机）→ 留一队列选 α + 块预言机汇总 + 六方法对比 + 参数量。
# 用法：bash drv_one.sh mstar [pathorchestra ...]
cd /path/to/systema4ST
say(){ echo "[$(date +%H:%M:%S)] $*"; }
E2="$*"; N=$(echo $E2 | wc -w); [ "$N" -ge 1 ] || { echo "无编码器"; exit 1; }
TAG=$(echo $E2 | tr ' ' '_' | cut -c1-20)
cat > jobs/emb_${TAG}.sh <<EOF
#!/bin/bash
#SBATCH -J emb_${TAG}
#SBATCH --qos=YOUR_QOS --partition=hpg-b200,hpg-rtx6000,hpg-turin --gres=gpu:1 --array=0-$((N-1)) -c 6 --mem=16G -t 24:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%A_%a.out
set -eu
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST; export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
export HF_TOKEN=\$(cat /path/to/.cache/huggingface/token); export HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN; unset HF_HUB_OFFLINE
A=($E2); X=\${A[\$SLURM_ARRAY_TASK_ID]}
[ "\$(ls results/hest_emb/*_\${X}.npz 2>/dev/null | wc -l)" -ge 72 ] && exit 0
python -u src/hest_embed_v2.py --encoder "\$X" --batch 128
EOF
J1=$(sbatch --parsable jobs/emb_${TAG}.sh); say "抽嵌入数组 $J1 已发（$E2）"
while squeue -j $J1 -h 2>/dev/null | grep -q .; do sleep 120; done
OK=""; for e in $E2; do [ "$(ls results/hest_emb/*_${e}.npz 2>/dev/null | wc -l)" -ge 72 ] && OK="$OK $e"; done
OK=$(echo $OK); NOK=$(echo $OK | wc -w); say "嵌入齐全 $NOK/$N: $OK"; [ "$NOK" -gt 0 ] || exit 1
ALPHAS="0.1 1 10 100 1000 10000 100000 1000000 10000000 100000000 1000000000"
cat > jobs/ds_${TAG}.sh <<EOF
#!/bin/bash
#SBATCH -J ds_${TAG}
#SBATCH --qos=YOUR_QOS --partition=hpg-default --array=0-$((NOK*13-1)) -c 4 --mem=24G -t 8:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%A_%a.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST; export OMP_NUM_THREADS=4 BLK_K=20,50,200
A=($OK); AL=($ALPHAS); I=\$SLURM_ARRAY_TASK_ID; X=\${A[\$((I/13))]}; J=\$((I%13))
if [ \$J = 0 ]; then O=results/hest_effres_ps_\${X}.json; [ -s \$O ] && exit 0; exec python -u src/hest_effres_ps.py --encoder \$X --skip_sigma --out \$O; fi
if [ \$J = 12 ]; then [ -s results/hest_blocks_\${X}.json ] && exit 0; exec python -u hest_blocks.py \$X; fi
a=\${AL[\$((J-1))]}; O=results/ridge_alpha/hest_ra_\${X}_a\${a}.json; [ -s \$O ] && exit 0
python -u src/hest_effres_ps.py --encoder \$X --skip_sigma --ridge_alpha \$a --out \$O
EOF
J2=$(sbatch --parsable jobs/ds_${TAG}.sh); say "下游数组 $J2 已发（$((NOK*13)) 任务）"
while squeue -j $J2 -h 2>/dev/null | grep -q .; do sleep 60; done
J3=$(sbatch --parsable --qos=YOUR_QOS -p hpg-default -c 2 --mem=16G -t 1:00:00 -J agg_${TAG} -o logs/agg_${TAG}_%j.out --wrap="source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest; cd /path/to/systema4ST; python3 ridge_loco.py $(echo $OK | tr ' ' ',') && python3 blk_agg.py | tail -8 && python3 methods_agg2.py | tail -12 && python3 pcount3.py $OK && S4ST_RESULTS=results python3 headline_numbers.py")
while squeue -j $J3 -h 2>/dev/null | grep -q .; do sleep 20; done
say "汇总完成"; grep -v "^$" logs/agg_${TAG}_${J3}.out | tail -40
