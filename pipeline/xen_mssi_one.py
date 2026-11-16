# -*- coding: utf-8 -*-
"""单个预测的 MSSI（作者发布的 mssi 包，mssi 环境）。由 xen_altrank.py 以子进程调用；粗化算子来自 xen_mssi.py。"""
import argparse, json, pickle
import numpy as np
# graph-coarsening (used by mssi) still calls np.bool / np.int / np.float, which numpy 1.24 removed;
# they were plain aliases of the Python builtins, so restoring them changes no computation.
for _a, _t in (("bool", bool), ("int", int), ("float", float)):
    if not hasattr(np, _a): setattr(np, _a, _t)
import pandas as pd
import mssi

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True); ap.add_argument("--ops", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
Z = np.load(a.npz); ops = pickle.load(open(a.ops, "rb"))
cols = [f"g{j}" for j in range(Z["truth"].shape[1])]
m = mssi.MSSI(pd.DataFrame(Z["truth"].astype(np.float64), columns=cols),
              pd.DataFrame(Z["pred"].astype(np.float64), columns=cols), ops).loc["MSSI"].to_numpy(dtype=float)
json.dump({"mssi": float(np.nanmean(m)), "nan_genes": int(np.isnan(m).sum())}, open(a.out, "w"))
