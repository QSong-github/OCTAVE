#!/bin/bash
#SBATCH --job-name=mkenv3
#SBATCH --qos=YOUR_QOS
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/project/logs/%x_%j.out
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
V=/path/to/project/venv_stmethods
M=/path/to/project/methods
source $V/bin/activate
echo "节点 $(hostname)"
echo "=== 补装 mkenv2 暴露的缺口 ==="
python -m pip install -q --no-input scprep easydl torch-geometric wandb 2>&1 | tail -6
echo "=== 复测受阻的三个方法 ==="
for k in HisToGene Hist2ST THItoGene; do
  echo "######## $k ########"
  python - "$M/$k" <<'PY' 2>&1 | head -6
import sys, os, glob, importlib.util
d=sys.argv[1]; sys.path.insert(0,d); ok,bad=[],[]
for p in sorted(glob.glob(os.path.join(d,"*.py"))):
    n=os.path.basename(p)[:-3]
    if n=="setup": continue
    try:
        s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); ok.append(n)
    except Exception as e: bad.append(f"{n} -> {type(e).__name__}: {str(e)[:55]}")
print("  ✅", ",".join(ok) if ok else "(无)")
for b in bad[:4]: print("  ❌", b)
PY
done
echo "=== 子包式方法: 按包导入 ==="
for k in STFlow:stflow STNet:stnet DeepSpot:.; do
  d="$M/${k%%:*}"; pkg="${k##*:}"
  echo "######## ${k%%:*} (包 $pkg) ########"
  ( cd "$d" && python -c "
import sys, importlib, glob, os
sys.path.insert(0,'.')
pkg='$pkg'
if pkg=='.':
    print('  顶层目录:', [x for x in os.listdir('.') if os.path.isdir(x)][:10])
    cands=[x for x in os.listdir('.') if os.path.isdir(x) and os.path.exists(os.path.join(x,'__init__.py'))]
    print('  可导入包:', cands)
    for c in cands[:3]:
        try: importlib.import_module(c); print('  ✅',c)
        except Exception as e: print('  ❌',c,type(e).__name__,str(e)[:55])
else:
    try:
        m=importlib.import_module(pkg); print('  ✅',pkg,'->',[x for x in dir(m) if not x.startswith('_')][:8])
    except Exception as e: print('  ❌',pkg,type(e).__name__,str(e)[:70])
" 2>&1 | head -8 )
done
echo "######## MKENV3 DONE ########"
