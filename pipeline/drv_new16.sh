#!/bin/bash
cd /blue/qsong1/wang.qing/systema4ST
say(){ echo "[$(date +%H:%M:%S)] $*"; }
python3 add_new15.py || exit 1
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
python3 -c "import sys; sys.path.insert(0,'src'); import hest_embed_v2 as h; print('ALL_ENC', len(h.ALL_ENC))" || exit 1
E="kaiko_vitb8 lunit_vits16 lunit_r50_swav lunit_r50_bt lunit_r50_moco ctranspath gpfm retccl dinov2_base dinov2_giant dinov3_vitb16 dinov3_vith16 clip_vitl14 pathgen_clip siglip2 biomedclip"
N=$(echo $E | wc -w)
say "阶段1 抽嵌入 $N 个（GPU）"
cat > jobs/new16emb.sh <<SH
#!/bin/bash
#SBATCH -J new16emb
#SBATCH --qos=qsong1 --partition=hpg-b200,hpg-rtx6000,hpg-turin
#SBATCH --gres=gpu:1 --array=0-$((N-1)) -c 6 --mem=16G -t 24:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%A_%a.out
set -eu
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 HF_TOKEN=\$(cat /blue/qsong1/wang.qing/.cache/huggingface/token)
export HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN
A=($E); X=\${A[\$SLURM_ARRAY_TASK_ID]}
[ "\$(ls results/hest_emb/*_\${X}.npz 2>/dev/null | wc -l)" -ge 72 ] && exit 0
echo "节点 \$(hostname) 编码器 \$X"
python -u src/hest_embed_v2.py --encoder "\$X" --batch 128
SH
sbatch jobs/new16emb.sh >/dev/null
while [ "$(squeue -u wang.qing -r -h -n new16emb | wc -l)" -gt 0 ]; do sleep 90; done
OK=""; for e in $E; do n=$(ls results/hest_emb/*_${e}.npz 2>/dev/null | wc -l); echo "  $e: $n/72"; [ "$n" -ge 72 ] && OK="$OK $e"; done
NOK=$(echo $OK | wc -w); say "嵌入齐全 $NOK/$N: $OK"
[ "$NOK" -gt 0 ] || exit 1
say "阶段2 下游：官方 ridge + 11 档 α + 块预言机（每编码器 13 个 CPU 任务）"
ALPHAS="0.1 1 10 100 1000 10000 100000 1000000 10000000 100000000 1000000000"
cat > jobs/new16ds.sh <<SH
#!/bin/bash
#SBATCH -J new16ds
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
sbatch jobs/new16ds.sh >/dev/null
while [ "$(squeue -u wang.qing -r -h -n new16ds | wc -l)" -gt 0 ]; do sleep 60; done
for e in $OK; do echo "  $e: ps=$([ -s results/hest_effres_ps_$e.json ] && echo 1 || echo 0) α=$(ls results/ridge_alpha/hest_ra_${e}_a*.json 2>/dev/null|wc -l)/11 blocks=$([ -s results/hest_blocks_$e.json ] && echo 1 || echo 0)"; done
say "阶段3 留一队列选 α + 块预言机汇总"
cat > jobs/new16agg.sh <<SH
#!/bin/bash
#SBATCH -J new16agg
#SBATCH --qos=qsong1 --partition=hpg-default -c 2 --mem=16G -t 1:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
python3 ridge_loco.py $(echo $OK | tr ' ' ',')
python3 blk_agg.py | tail -12
SH
J=$(sbatch --parsable jobs/new16agg.sh); while [ "$(squeue -j $J -h | wc -l)" -gt 0 ]; do sleep 20; done; cat logs/new16agg_${J}.out
