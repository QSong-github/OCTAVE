#!/bin/bash
#SBATCH --job-name=stflow7
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%j.out
# 最后两处: ① 两脚本都要父目录(我先前读错了 benchmark.py 的代码路径)
#          ② scvi 是废弃包, 需要 scvi-tools 才有 scvi.distributions(依赖扫描器映射表的漏洞)
set -u
V=/path/to/systema4ST/venv_np1
M=/path/to/systema4ST/methods/STFlow
B=/path/to/he2st/HEST/eval/bench_data
OUT=/path/to/systema4ST/stflow_run
export HF_HOME=/path/to/systema4ST/.hf
export CHECKPOINT_PATH=$OUT/weights/fm_v1/ciga/tenpercent_resnet18.ckpt
source $V/bin/activate
echo "节点 $(hostname)"

echo "=== 换 scvi → scvi-tools ==="
python -m pip uninstall -y -q scvi 2>&1 | tail -2
python -m pip install -q --no-input scvi-tools 2>&1 | tail -3
python -c "from scvi.distributions import ZeroInflatedNegativeBinomial; print('  ✅ scvi.distributions 可用')" || exit 1

ALL="CCRCC COAD HCC IDC LUNG LYMPH_IDC PAAD PRAD READ SKCM"
echo "=== 全队列特征提取(source_dataroot=父目录) ==="
cd $M/stflow/app/hest
for C in $ALL; do
  n=$(find $OUT/embed/$C -name "*.h5" 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && { echo "  $C 已有 $n 个, 跳过"; continue; }
  echo "---- $C ----"
  python -u benchmark.py --datasets $C --encoders ciga --source_dataroot $B \
    --embed_dataroot $OUT/embed --weights_root $OUT/weights --results_dir $OUT/results \
    --exp_code ciga_$C --batch_size 128 2>&1 | tr '\r' '\n' | grep -avE "%\|" | tail -5
done
TOT=$(find $OUT/embed -name '*.h5' | wc -l)
echo "  嵌入总数 = $TOT (应为 72)"
[ "$TOT" -lt 10 ] && { echo "❌ 嵌入仍不足, 停止"; exit 1; }

echo "=== 训练冒烟 ==="
cd $M/stflow/app/flow
python -u train.py --datasets SKCM --feature_encoder ciga --source_dataroot $B \
  --embed_dataroot $OUT/embed --save_dir $OUT/results --exp_code smoke3_SKCM \
  --epochs 5 --batch_size 2 --num_workers 4 2>&1 | tr '\r' '\n' | grep -avE "%\|" | tail -18
ls -d $OUT/results/smoke3_SKCM* >/dev/null 2>&1 || { echo "❌ 冒烟未产出结果目录, 停止"; exit 1; }

echo "=== 全队列训练 100 epoch ==="
for C in $ALL; do
  echo "######## $C ########"
  python -u train.py --datasets $C --feature_encoder ciga --source_dataroot $B \
    --embed_dataroot $OUT/embed --save_dir $OUT/results --exp_code full_$C \
    --epochs 100 --batch_size 2 --num_workers 4 2>&1 | tr '\r' '\n' | grep -avE "%\|" | tail -8
done
echo "######## STFLOW7 DONE ########"
