#!/bin/bash
#SBATCH -J tpxenv4
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 2 --mem=16G -t 2:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
# PyPI 的 "hest" 是别的项目；MahmoodLab 的 HEST 只能从 GitHub 装。装到独立目录（不动 he2st/HEST 旧版），置于 PYTHONPATH 最前。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate triplex
cd /path/to/project
pip install --no-deps --target methods/hest_new "git+https://github.com/mahmoodlab/HEST.git" 2>&1 | tail -2
ls methods/hest_new | head; ls methods/hest_new/hest/bench 2>/dev/null | head
export PYTHONPATH=/path/to/project/methods/hest_new:/path/to/project/methods/triplex_shim:/path/to/project/methods/TRIPLEX/src
python - <<PY
import importlib
for m in ["hest","hest.HESTData","hest.bench.cpath_model_zoo.inference_models","hest.bench.cpath_model_zoo.utils.transform_utils","hest.bench.utils.file_utils","hestcore.wsi"]:
    try: mod=importlib.import_module(m); print("OK ", m, getattr(mod,"__file__",""))
    except Exception as e: print("缺 ", m, type(e).__name__, str(e)[:200])
try:
    from hest.HESTData import read_HESTData; from hest import iter_hest; print("read_HESTData/iter_hest OK")
except Exception as e: print("read_HESTData:", type(e).__name__, str(e)[:160])
try:
    from hest.bench.cpath_model_zoo.inference_models import inf_encoder_factory; import inspect
    cls=inf_encoder_factory("uni_v1"); print("uni_v1 loader:", cls, inspect.signature(cls.__init__)); src=inspect.getsource(cls); print(src[:1500])
except Exception as e: print("inf_encoder_factory:", type(e).__name__, str(e)[:200])
import os; os.chdir("methods/TRIPLEX")
for m in ["preprocess.prepare_data","preprocess.extract_img_features"]:
    try: importlib.import_module(m); print("OK  TRIPLEX", m)
    except SystemExit: print("OK  TRIPLEX", m, "(argparse SystemExit)")
    except Exception as e: print("缺  TRIPLEX", m, type(e).__name__, str(e)[:200])
PY
