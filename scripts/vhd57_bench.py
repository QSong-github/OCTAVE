# -*- coding: utf-8 -*-
"""Visium HD：57 个编码器在 2 个结肠切片上的 PCC 与最细带分数（统一汇总规则：每个切片一个值，跨 2 个切片取中位数）。
输入 results/blocks_vhd_bands_{编码器}/{切片}.json（pred.ridge / pred.dom20 的 pcc 与 band_pcc，band_pcc 为最细带）。"""
import json, glob, itertools, os, sys
import numpy as np
from scipy.stats import spearmanr
R = os.environ.get("OCTAVE_RESULTS", "results"); rows = {}
for d in sorted(glob.glob(f"{R}/blocks_vhd_bands_*")):
    X = [json.load(open(f)) for f in sorted(glob.glob(d + "/*.json"))]; assert len(X) == 2, d
    g = lambda p, k: float(np.median([x["pred"][p][k] for x in X]))
    rows[d.split("blocks_vhd_bands_")[1]] = dict(pcc=g("ridge", "pcc"), b1=g("ridge", "band_pcc"), opcc=g("dom20", "pcc"), ob1=g("dom20", "band_pcc"))
E = sorted(rows); assert len(E) == 57
P = np.array([rows[e]["pcc"] for e in E]); B = np.array([rows[e]["b1"] for e in E])
O = np.array([rows[e]["opcc"] for e in E]); OB = np.array([rows[e]["ob1"] for e in E])
rev = sum((P[i] - P[j]) * (B[i] - B[j]) < 0 for i, j in itertools.combinations(range(57), 2))
r3 = lambda v: round(float(v), 3)
print(f"PCC {r3(P.min())}-{r3(P.max())} factor {r3(P.max())/r3(P.min()):.2f}; finest {r3(B.min())}-{r3(B.max())} factor {r3(B.max())/r3(B.min()):.2f}")
print(f"Spearman {spearmanr(P, B).statistic:.3f}; reversed {rev} of 1596; best {E[P.argmax()]} / {E[B.argmax()]}; worst {E[P.argmin()]} / {E[B.argmin()]}")
print(f"oracle below model for {(O < P).sum()} (PCC) and {(OB < B).sum()} (finest); medians {np.median(O):.3f} vs {np.median(P):.3f}, {np.median(OB):.3f} vs {np.median(B):.3f}")
json.dump(rows, open(os.environ.get("OUT", f"{R}/vhd57_bench.json"), "w"), indent=1)
