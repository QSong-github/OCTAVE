#!/bin/bash
#SBATCH --job-name=stflow6
#SBATCH --qos=qsong1
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
# 三处修正:
#  ① 两脚本路径约定相反: benchmark.py 要队列目录, train.py:147 要父目录(它自己再拼 dataset)
#  ② STFlow 发布代码(commit 880c2ee)自身 API 不一致: transformer.py:160 给 GeneUpdate 传
#     non_negative=, 而 GeneUpdate.__init__ 没这个参数 → 默认 backbone 构造即崩。
#     仓库只有一个 commit、一个分支, 无其它版本可用。最小改动: 去掉这个【当前实现里
#     根本不存在的】关键字, 不新增任何行为。改动留痕, 论文需在补充材料说明。
#  ③ 先诊断为何只有 HCC/SKCM 出了嵌入
set -u
V=/blue/qsong1/wang.qing/systema4ST/venv_np1
M=/blue/qsong1/wang.qing/systema4ST/methods/STFlow
B=/blue/qsong1/wang.qing/he2st/HEST/eval/bench_data
OUT=/blue/qsong1/wang.qing/systema4ST/stflow_run
export HF_HOME=/blue/qsong1/wang.qing/systema4ST/.hf
export CHECKPOINT_PATH=$OUT/weights/fm_v1/ciga/tenpercent_resnet18.ckpt
source $V/bin/activate
echo "节点 $(hostname)"

echo "=== ① GeneUpdate 是否真的用到非负性 ==="
sed -n "19,45p" $M/stflow/model/transformer.py

echo "=== ② 打补丁 ==="
python - <<'PY'
p="/blue/qsong1/wang.qing/systema4ST/methods/STFlow/stflow/model/transformer.py"
s=open(p).read()
old="GeneUpdate(d_model, n_genes, proj_drop=proj_drop, non_negative=gene_exp_non_negative)"
new="GeneUpdate(d_model, n_genes, proj_drop=proj_drop)  # [systema4ST] 去掉 GeneUpdate.__init__ 不接受的 non_negative"
if old in s:
    open(p,"w").write(s.replace(old,new)); print("  ✅ 已修正 GeneUpdate 调用")
else:
    print("  已处理过或调用点已变")
PY

echo "=== ③ 诊断一个未出嵌入的队列(READ) ==="
cd $M/stflow/app/hest
python -u benchmark.py --datasets READ --encoders ciga --source_dataroot $B/READ \
  --embed_dataroot $OUT/embed --weights_root $OUT/weights --results_dir $OUT/results \
  --exp_code diag_READ --batch_size 128 2>&1 | tr '\r' '\n' | grep -avE "%\|" | tail -20

echo "=== ④ 全队列特征提取 ==="
for C in CCRCC COAD HCC IDC LUNG LYMPH_IDC PAAD PRAD READ SKCM; do
  n=$(find $OUT/embed/$C -name "*.h5" 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && { echo "  $C 已有 $n 个, 跳过"; continue; }
  echo "---- $C ----"
  python -u benchmark.py --datasets $C --encoders ciga --source_dataroot $B/$C \
    --embed_dataroot $OUT/embed --weights_root $OUT/weights --results_dir $OUT/results \
    --exp_code ciga_$C --batch_size 128 2>&1 | tr '\r' '\n' | grep -avE "%\|" | tail -6
done
echo "  嵌入总数: $(find $OUT/embed -name '*.h5' | wc -l)"

echo "=== ⑤ 训练冒烟(source_dataroot 改父目录) ==="
cd $M/stflow/app/flow
python -u train.py --datasets SKCM --feature_encoder ciga \
  --source_dataroot $B --embed_dataroot $OUT/embed --save_dir $OUT/results \
  --exp_code smoke2_SKCM --epochs 5 --batch_size 2 --num_workers 4 2>&1 | tr '\r' '\n' | grep -avE "%\|" | tail -20
echo "######## STFLOW6 DONE ########"
