# -*- coding: utf-8 -*-
"""用作者发布的 mssi 包（Haviv et al.）给 xen_altmetrics.py 存下的同一组预测打 MSSI 分。
在 mssi 环境里跑（该包锁 scipy==1.11.0）。除官方总分外，按官方循环逐尺度记下 l / c / s 三项，便于逐尺度比较。
粗化算子按区域存盘，编码器排序实验直接复用。"""
import argparse, json, os, pickle
import numpy as np
# graph-coarsening (used by mssi) still calls np.bool / np.int / np.float, which numpy 1.24 removed;
# they were plain aliases of the Python builtins, so restoring them changes no computation.
for _a, _t in (("bool", bool), ("int", int), ("float", float)):
    if not hasattr(np, _a): setattr(np, _a, _t)
import pandas as pd
import mssi
from mssi.utils import scale_max, cal_ssim
from graph_coarsening.coarsening_utils import coarsen_vector
from scipy import sparse as _sp
import graph_coarsening.graph_utils as _gu


def _get_S_sparse(G):
    """graph_utils.get_S 的稀疏写法：同一个 N×|E| 梯度矩阵（每条边两个非零元 ±sqrt(w)），只是不分配稠密数组。
    mssi.ms_coars 调用它后并不使用结果；原函数在 19 万 bin 以上的区域要申请数 TB，因此替换。"""
    e = G.get_edge_list(); w = np.sqrt(np.asarray(e[2], float)); i, j = np.asarray(e[0]), np.asarray(e[1]); M = len(w)
    return _sp.csr_matrix((np.concatenate([w, -w]), (np.concatenate([i, j]), np.concatenate([np.arange(M)] * 2))),
                          shape=(G.N, M))


_gu.get_S = _get_S_sparse

ROOT = "/path/to/project"
NPZD, OPSD, OUTD = f"{ROOT}/results/xen_altmetrics_npz", f"{ROOT}/results/xen_mssi_ops", f"{ROOT}/results/xen_mssi"
COEF = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]


class _ST:
    def __init__(self, xy):
        self.obsm = {"spatial": xy}; self.obs = pd.DataFrame(index=np.arange(len(xy)))


def per_scale(T, Pm, ops):
    """与 mssi.MSSI 同一循环，返回逐尺度逐基因的 (l, c, s)。"""
    T, Pm = np.asarray(scale_max(T)), np.asarray(scale_max(Pm)); out = []
    for ind in range(len(COEF)):
        l = np.empty(T.shape[1]); c = np.empty(T.shape[1]); s = np.empty(T.shape[1])
        for j in range(T.shape[1]):
            l[j], c[j], s[j] = cal_ssim(T[:, j], Pm[:, j], max(T[:, j].max(), Pm[:, j].max()))
        out.append((l, c, s))
        if ind < len(COEF) - 1:
            T, Pm = coarsen_vector(T, ops[0][ind]), coarsen_vector(Pm, ops[0][ind])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); a = ap.parse_args()
    Z = np.load(f"{NPZD}/{a.name}.npz", allow_pickle=False)
    xy = Z["xy"].astype(np.float64); genes = [str(g) for g in Z["genes"]]
    os.makedirs(OPSD, exist_ok=True); os.makedirs(OUTD, exist_ok=True)
    opf = f"{OPSD}/{a.name}.pkl"
    if os.path.exists(opf):
        ops = pickle.load(open(opf, "rb"))
    else:
        ops, graphs = mssi.ms_coars(_ST(xy), n_scales=len(COEF))
        pickle.dump(ops, open(opf, "wb"))
        print(f"[{a.name}] 粗化各尺度节点数 {[g.N for g in graphs[0]]}", flush=True)
    truth = pd.DataFrame(Z["truth"].astype(np.float64), columns=genes)
    res = {"name": a.name, "n": int(len(xy)), "pred": {}}
    for nm in ("ridge", "dom20", "blur", "white_noise", "boundary_shift", "hotspot_removal"):
        P = pd.DataFrame(Z[nm].astype(np.float64), columns=genes)
        m = mssi.MSSI(truth, P, ops).loc["MSSI"].to_numpy(dtype=float)
        sc = per_scale(truth, P, ops)
        res["pred"][nm] = {"mssi": float(np.nanmean(m)), "mssi_nan_genes": int(np.isnan(m).sum()),
                           "per_scale": {str(i + 1): {"l": float(np.nanmean(l)), "c": float(np.nanmean(c)),
                                                      "s": float(np.nanmean(s))} for i, (l, c, s) in enumerate(sc)}}
        print(f"  {nm:<16s} MSSI {res['pred'][nm]['mssi']:.4f}  s1..s5 "
              + " ".join(f"{res['pred'][nm]['per_scale'][str(i)]['s']:.3f}" for i in range(1, 6)), flush=True)
    json.dump(res, open(f"{OUTD}/{a.name}.json", "w"), indent=1)
    print(f"→ {OUTD}/{a.name}.json", flush=True)


if __name__ == "__main__":
    main()
