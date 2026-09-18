#!/bin/bash
#SBATCH -J tpxenv5
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 2 --mem=16G -t 2:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%j.out
# HEST 的旧模型库在 2026-03 被移除（#132）；TRIPLEX（2025-05）需要它。装 v1.2.0，不行再退到 v1.1.0。检查失败则本作业失败（下游依赖 afterok）。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate triplex
cd /path/to/systema4ST
export PYTHONPATH=/path/to/systema4ST/methods/hest_new:/path/to/systema4ST/methods/triplex_shim:/path/to/systema4ST/methods/TRIPLEX/src
check() { python - <<PY
import importlib, sys, os
bad=0
for m in ["hest.bench.cpath_model_zoo.inference_models","hest.bench.cpath_model_zoo.utils.transform_utils","hest.bench.utils.file_utils","hestcore.wsi","hestcore.segmentation"]:
    try: mod=importlib.import_module(m); print("OK ", m, mod.__file__)
    except Exception as e: print("缺 ", m, type(e).__name__, str(e)[:160]); bad+=1
try:
    from hest.HESTData import read_HESTData; from hest import iter_hest; print("read_HESTData/iter_hest OK")
except Exception as e: print("read_HESTData:", type(e).__name__, str(e)[:160]); bad+=1
try:
    from hest.bench.cpath_model_zoo.inference_models import inf_encoder_factory; import inspect
    cls=inf_encoder_factory("uni_v1"); print("uni_v1 loader:", cls.__name__, inspect.signature(cls.__init__)); print(inspect.getsource(cls)[:900])
except Exception as e: print("inf_encoder_factory:", type(e).__name__, str(e)[:160]); bad+=1
os.chdir("methods/TRIPLEX")
for m in ["preprocess.prepare_data","preprocess.extract_img_features","dataset","model","utils.train_utils"]:
    try: importlib.import_module(m); print("OK  TRIPLEX", m)
    except SystemExit: print("OK  TRIPLEX", m)
    except Exception as e: print("缺  TRIPLEX", m, type(e).__name__, str(e)[:160]); bad+=1
sys.exit(1 if bad else 0)
PY
}
for tag in v1.2.0 v1.1.0; do
  rm -rf methods/hest_new; pip install --no-deps --target methods/hest_new "git+https://github.com/mahmoodlab/HEST.git@$tag" 2>&1 | tail -1
  echo "== 试 $tag =="; if check; then echo "HEST $tag 可用"; exit 0; fi
done
echo "两个版本都不行"; exit 1
