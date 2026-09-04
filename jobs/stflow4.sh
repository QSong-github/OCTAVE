#!/bin/bash
#SBATCH --job-name=stflow4
#SBATCH --qos=qsong1
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
# CIGA 权重已下好(92MB, 有效 Lightning ckpt), 只是 STFlow 期望 weights/fm_v1/ciga/。
# 本作业自带闸门: 冒烟产出嵌入才铺开全部队列, 否则立即停, 不烧 GPU。
set -u
V=/blue/qsong1/wang.qing/systema4ST/venv_np1
M=/blue/qsong1/wang.qing/systema4ST/methods/STFlow
B=/blue/qsong1/wang.qing/he2st/HEST/eval/bench_data
OUT=/blue/qsong1/wang.qing/systema4ST/stflow_run
export HF_HOME=/blue/qsong1/wang.qing/systema4ST/.hf
source $V/bin/activate
echo "节点 $(hostname)"

mkdir -p $OUT/weights/fm_v1/ciga
ln -f $OUT/weights/ciga/tenpercent_resnet18.ckpt $OUT/weights/fm_v1/ciga/tenpercent_resnet18.ckpt
export CHECKPOINT_PATH=$OUT/weights/fm_v1/ciga/tenpercent_resnet18.ckpt
ls -la $OUT/weights/fm_v1/ciga/

run_extract () { cd $M/stflow/app/hest; python -u benchmark.py --datasets $1 --encoders ciga \
    --source_dataroot $B/$1 --embed_dataroot $OUT/embed --weights_root $OUT/weights \
    --results_dir $OUT/results --exp_code ciga_$1 --batch_size 128 2>&1 | tr '\r' '\n' | grep -avE "%\|" | tail -8; }
run_train () { cd $M/stflow/app/flow; python -u train.py --datasets $1 --feature_encoder ciga \
    --source_dataroot $B/$1 --embed_dataroot $OUT/embed --save_dir $OUT/results \
    --exp_code $2_$1 --epochs $3 --batch_size 2 --num_workers 4 2>&1 | tr '\r' '\n' | grep -avE "%\|" | tail -14; }

echo "======== 冒烟: SKCM ========"
run_extract SKCM
N=$(find $OUT/embed/SKCM -name "*.h5" 2>/dev/null | wc -l)
echo "  SKCM 嵌入文件数 = $N"
if [ "$N" -lt 1 ]; then echo "❌ 冒烟未产出嵌入, 停止(不铺开)"; exit 1; fi
run_train SKCM smoke 10

echo "======== 闸门通过, 铺开全部队列 ========"
for C in CCRCC COAD HCC IDC LUNG LYMPH_IDC PAAD PRAD READ SKCM; do
  echo "######## $C 特征提取 ########"; run_extract $C
done
for C in CCRCC COAD HCC IDC LUNG LYMPH_IDC PAAD PRAD READ SKCM; do
  echo "######## $C 训练(100 epoch) ########"; run_train $C full 100
done
echo "######## STFLOW4 DONE ########"
