#!/bin/bash
#SBATCH --job-name=stflow3
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%j.out
# STFlow 走 ciga 路线: UNI/GigaPath 都是 HF gated, 但 train.py 的 feature_dim 里
# 第三个选项 ciga(512d) 对应 Ciga 等人的自监督病理 ResNet18, 权重在 GitHub release, 开放。
#
# 对第三方代码只做一处改动: 注释掉 benchmark.py 里【与所选编码器无关的】UNI/GigaPath
# 无条件下载(第 253-256 行, 属安装脚本残留)。方法本身逻辑一行未动, 改动留痕以备复核。
set -u
V=/path/to/systema4ST/venv_np1
M=/path/to/systema4ST/methods/STFlow
B=/path/to/he2st/HEST/eval/bench_data
OUT=/path/to/systema4ST/stflow_run
export HF_HOME=/path/to/systema4ST/.hf
source $V/bin/activate
echo "节点 $(hostname)"; mkdir -p $OUT/weights/ciga $OUT/embed $OUT/results

echo "=== 0. Hist2ST 的 collections.Iterable 兼容垫片(Py3.10 起已移除) ==="
SC=$V/lib/python3.11/site-packages/sitecustomize.py
grep -q "collections.Iterable" $SC 2>/dev/null || cat >> $SC <<'PY'
import collections, collections.abc
for _n in ("Iterable","Mapping","MutableMapping","Sequence","Callable","Hashable"):
    if not hasattr(collections, _n): setattr(collections, _n, getattr(collections.abc, _n))
PY
python -c "import collections; print('  ✅ collections.Iterable ->', collections.Iterable)"

echo "=== 1. 编码器配置在哪 ==="
grep -rn "ciga" $M/stflow/hest_utils/*.py $M/stflow/hest_utils/*.yaml $M/stflow/hest_utils/*.json 2>/dev/null | head -8
find $M -name "*.yaml" -o -name "*.json" | head -10

echo "=== 2. 下载 CIGA 权重(GitHub release, 无需认证) ==="
CK=$OUT/weights/ciga/tenpercent_resnet18.ckpt
if [ ! -s "$CK" ]; then
  for U in \
    "https://github.com/ozanciga/self-supervised-histopathology/releases/download/tenpercent/tenpercent_resnet18.ckpt" \
    "https://github.com/ozanciga/self-supervised-histopathology/releases/download/v1.0/tenpercent_resnet18.ckpt"; do
    echo "  尝试 $U"
    curl -sSL -m 600 -o "$CK" "$U" && [ -s "$CK" ] && break
  done
fi
ls -la "$CK" 2>/dev/null && python -c "
import torch,sys
try:
    d=torch.load('$CK',map_location='cpu',weights_only=False)
    print('  ✅ ckpt 可读, 顶层键:', list(d.keys())[:5] if isinstance(d,dict) else type(d))
except Exception as e: print('  ❌',e); sys.exit(1)" || echo "  ❌ CIGA 权重获取失败"

echo "=== 3. 去掉与所选编码器无关的无条件下载 ==="
python - <<PY
import re
p="$M/stflow/app/hest/benchmark.py"
s=open(p).read()
if "# [systema4ST]" not in s:
    for pat in ['snapshot_download(repo_id="MahmoodLab/hest-bench"','hf_hub_download("MahmoodLab/UNI"','hf_hub_download("prov-gigapath/prov-gigapath"']:
        s=re.sub(r'(\n\s*)('+re.escape(pat)+r')', r'\1# [systema4ST] 与 --encoders 无关的安装脚本残留, 已停用\n\1# \2', s)
    open(p,"w").write(s); print("  ✅ 已注释 3 处无条件下载")
else: print("  已处理过")
PY

echo "=== 4. STFlow 冒烟 (ciga) ==="
cd $M/stflow/app/hest
for C in SKCM HCC; do
  echo "---- 特征提取 $C ----"
  python -u benchmark.py --datasets $C --encoders ciga \
     --source_dataroot $B/$C --embed_dataroot $OUT/embed --weights_root $OUT/weights \
     --results_dir $OUT/results --exp_code ciga_$C --batch_size 128 2>&1 | tr '\r' '\n' | grep -avE "%\|" | tail -14
done
cd $M/stflow/app/flow
for C in SKCM HCC; do
  echo "---- 训练 $C ----"
  python -u train.py --datasets $C --feature_encoder ciga \
     --source_dataroot $B/$C --embed_dataroot $OUT/embed --save_dir $OUT/results \
     --exp_code smoke_$C --epochs 10 --batch_size 2 --num_workers 4 2>&1 | tr '\r' '\n' | grep -avE "%\|" | tail -20
done
echo "######## STFLOW3 DONE ########"
