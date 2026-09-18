#!/bin/bash
#SBATCH -J tpxenv2
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 2 --mem=16G -t 2:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%j.out
# TRIPLEX 需要新版 HEST（hest.bench.cpath_model_zoo、hestcore）。不动 he2st/HEST 的旧版：把新版 hest 装到独立目录并置于 PYTHONPATH 最前；hestcore 与 wget 装进 triplex 环境。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate triplex
cd /path/to/systema4ST
pip install wget 2>&1 | tail -1
pip index versions hest 2>/dev/null | head -2; pip index versions hestcore 2>/dev/null | head -2
pip install hestcore 2>&1 | tail -2
[ -d methods/hest_new/hest ] || pip install --no-deps --target methods/hest_new hest 2>&1 | tail -2
export PYTHONPATH=/path/to/systema4ST/methods/hest_new:/path/to/systema4ST/methods/triplex_shim:/path/to/systema4ST/methods/TRIPLEX/src
python - <<PY
import importlib, traceback
for m in ["hest","hestcore","hestcore.wsi","hestcore.segmentation","hest.bench.cpath_model_zoo.inference_models","hest.bench.cpath_model_zoo.utils.transform_utils","hest.bench.utils.file_utils","wget","flash_attn"]:
    try:
        mod=importlib.import_module(m); print("OK ", m, getattr(mod,"__version__",""), getattr(mod,"__file__",""))
    except Exception as e: print("缺 ", m, type(e).__name__, str(e)[:160])
try:
    from hest import iter_hest; from hest.HESTData import HESTData; import inspect; print("新 hest iter_hest OK; dump_patches:", str(inspect.signature(HESTData.dump_patches))[:150])
except Exception as e: print("iter_hest:", type(e).__name__, str(e)[:160])
import os; os.chdir("methods/TRIPLEX")
for m in ["preprocess.prepare_data","dataset.feature_dataset","dataset","model","utils.train_utils"]:
    try: importlib.import_module(m); print("OK  TRIPLEX", m)
    except Exception as e: print("缺  TRIPLEX", m, type(e).__name__, str(e)[:160])
PY
