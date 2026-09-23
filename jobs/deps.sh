#!/bin/bash
#SBATCH --job-name=deps
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/project/logs/%x_%j.out
# STFlow 的 setup.py 没有 install_requires, 逐个试错要烧很多轮。
# 改为扫出全部仓库的 import, 一次性解析并安装缺失的顶层模块, 然后直接冒烟。
set -u
V=/path/to/project/venv_np1
M=/path/to/project/methods
B=/path/to/he2st/HEST/eval/bench_data
OUT=/path/to/project/stflow_run
export HF_HOME=/path/to/project/.hf
source $V/bin/activate
echo "节点 $(hostname)"

echo "=== 1. 扫描全部方法仓库的 import ==="
python - "$M" <<'PY'
import sys, os, re, ast, subprocess, importlib.util
root = sys.argv[1]
STD = set(sys.stdlib_module_names)
# import 名 → pip 包名(不一致的)
ALIAS = {"cv2":"opencv-python-headless","PIL":"pillow","sklearn":"scikit-learn",
         "skimage":"scikit-image","yaml":"pyyaml","torch_geometric":"torch-geometric",
         "pytorch_lightning":"pytorch-lightning","huggingface_hub":"huggingface_hub",
         "openslide":"openslide-python","hydra":"hydra-core","omegaconf":"omegaconf",
         "attr":"attrs","dateutil":"python-dateutil","Bio":"biopython"}
LOCAL = set()          # 仓库内部模块, 不能当 pip 包
for d, _, fs in os.walk(root):
    for f in fs:
        if f.endswith(".py"): LOCAL.add(f[:-3])
    LOCAL.update(os.path.basename(d) for _ in [0])
mods = set()
for d, _, fs in os.walk(root):
    if any(x in d for x in (".git", "site-packages")): continue
    for f in fs:
        if not f.endswith(".py"): continue
        try: tree = ast.parse(open(os.path.join(d, f), encoding="utf-8", errors="ignore").read())
        except Exception: continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for al in n.names: mods.add(al.name.split(".")[0])
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                mods.add(n.module.split(".")[0])
cand = sorted(m for m in mods if m not in STD and m not in LOCAL and not m.startswith("_"))
missing = [m for m in cand if importlib.util.find_spec(m) is None]
print("  扫到第三方模块", len(cand), "个; 缺失", len(missing), "个:")
print("   ", ", ".join(missing))
pkgs = sorted({ALIAS.get(m, m) for m in missing})
print("  → pip 目标:", " ".join(pkgs))
for p in pkgs:
    r = subprocess.run([sys.executable,"-m","pip","install","-q","--no-input",p],
                       capture_output=True, text=True)
    print(("  ✅ " if r.returncode==0 else "  ❌ ")+p+("" if r.returncode==0 else " :: "+r.stderr.strip().splitlines()[-1][:80]))
PY

echo "=== 2. 复测各方法 import ==="
for k in STFlow BLEEP HisToGene Hist2ST THItoGene; do
  echo "-- $k --"
  python - "$M/$k" <<'PY' 2>&1 | head -4
import sys,os,glob,importlib.util
d=sys.argv[1]; sys.path.insert(0,d); ok,bad=[],[]
for p in sorted(glob.glob(os.path.join(d,"*.py")))+sorted(glob.glob(os.path.join(d,"*","__init__.py"))):
    n=os.path.basename(p)[:-3] if p.endswith(".py") and "__init__" not in p else os.path.basename(os.path.dirname(p))
    if n=="setup": continue
    try:
        s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); ok.append(n)
    except Exception as e: bad.append(f"{n}:{type(e).__name__}:{str(e)[:45]}")
print("  ✅",",".join(ok[:12]) if ok else "(无)")
for b in bad[:3]: print("  ❌",b)
PY
done

echo "=== 3. STFlow 冒烟 ==="
cd $M/stflow/app/hest 2>/dev/null || cd $M/STFlow/stflow/app/hest
for C in SKCM HCC; do
  echo "---- 特征提取 $C ----"
  python -u benchmark.py --datasets $C --encoders resnet50_trunc \
      --source_dataroot $B/$C --embed_dataroot $OUT/embed --weights_root $OUT/weights \
      --results_dir $OUT/results --exp_code r50_$C --batch_size 128 2>&1 | tr '\r' '\n' | grep -avE "%\|" | tail -12
done
cd $M/STFlow/stflow/app/flow
for C in SKCM HCC; do
  echo "---- 训练 $C ----"
  python -u train.py --datasets $C --feature_encoder resnet50_trunc \
      --source_dataroot $B/$C --embed_dataroot $OUT/embed --save_dir $OUT/results \
      --exp_code smoke_$C --epochs 10 --batch_size 2 --num_workers 4 2>&1 | tr '\r' '\n' | grep -avE "%\|" | tail -18
done
echo "######## DEPS DONE ########"
