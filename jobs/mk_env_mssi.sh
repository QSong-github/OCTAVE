#!/bin/bash
#SBATCH -J mkmssi
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 2 --mem=48G --nice=50000 -t 2:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
# a separate env for the published MSSI package (Haviv et al.), which pins scipy==1.11.0.
# The shared hest env is not touched.
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
E=/path/to/miniconda3/envs/mssi
if [ ! -d $E ]; then
  conda create -y -n mssi -c conda-forge python=3.11 "numpy=1.26" "scipy=1.11.0" "pandas=2.1" "scikit-learn=1.3" matplotlib sortedcontainers networkx
fi
conda activate mssi
pip install --no-cache-dir --no-deps pygsp==0.5.1 mssi==0.1.9 "git+https://github.com/loukasa/graph-coarsening"
# keep numpy 1.26 (matplotlib needs >=1.25); the removed np.bool/np.int/np.float aliases are restored in our scripts instead
pip install --no-cache-dir --no-deps "numpy==1.26.4"
python - <<'PY'
import numpy as np
# graph-coarsening (used by mssi) still calls np.bool / np.int / np.float, which numpy 1.24 removed;
# they were plain aliases of the Python builtins, so restoring them changes no computation.
for _a, _t in (("bool", bool), ("int", int), ("float", float)):
    if not hasattr(np, _a): setattr(np, _a, _t)
import pandas as pd, scipy
import mssi, graph_coarsening, pygsp
print("scipy", scipy.__version__, "numpy", np.__version__)
class A: pass
rng = np.random.default_rng(0)
n = 3000
xy = rng.uniform(0, 1000, (n, 2))
truth = np.stack([np.sin(xy[:, 0] / (50 * (i + 1))) + 0.3 * rng.normal(size=n) for i in range(4)], 1)
pred = truth + rng.normal(scale=0.5, size=truth.shape)
st = A(); st.obsm = {"spatial": xy}; st.obs = pd.DataFrame(index=range(n))
ops, graphs = mssi.ms_coars(st, n_scales=5)
print("scales:", [g.N for g in graphs[0]])
cols = [f"g{i}" for i in range(4)]
print(mssi.MSSI(pd.DataFrame(truth, columns=cols), pd.DataFrame(pred, columns=cols), ops))
print("SMOKE OK")
PY
