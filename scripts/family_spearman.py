# -*- coding: utf-8 -*-
"""参数量与预言机比值的 Spearman 相关，编码器层面与家族层面（正文 §4 引用的四个数）。
家族把同一配方或同一教师的编码器合并为一个点，家族值取其成员的中位数。
输出 results/family_spearman.json。"""
import json, os, collections, numpy as np

R = os.environ.get("OCTAVE_RESULTS", "results")
S = {e["enc"]: e for e in json.load(open(f"{R}/hest_blocks_summary.json"))}
P = json.load(open(f"{R}/encoder_params.json"))
DISTILLED = {"litevirchow2", "h0_mini", "litefm", "litefm_l", "litefm_s",
             "distillpath_is16", "distillpath_ks16", "pathryoshka_b"}
GROUPS = {"uni": ["uni_v1", "uni_v2"], "virchow": ["virchow", "virchow2", "litevirchow2"],
          "hoptimus": ["hoptimus0", "hoptimus1", "h0_mini"], "phikon": ["phikon", "phikon_v2"],
          "conch": ["conch_v1", "conch_v15"], "gigapath": ["gigapath", "gigapath_flash"],
          "hibou": ["hibou_b", "hibou_l"],
          "kaiko": ["kaiko_vitb16", "kaiko_vitb8", "kaiko_vitl14", "kaiko_vits16", "midnight12k", "openmidnight"],
          "lunit": ["lunit_r50_bt", "lunit_r50_moco", "lunit_r50_swav", "lunit_vits16", "lunit_vits8"],
          "dinov2": ["dinov2_base", "dinov2_large", "dinov2_giant"],
          "dinov3": ["dinov3_vitb16", "dinov3_vith16", "dinov3_vitl16"],
          "litefm": ["litefm", "litefm_l", "litefm_s"], "distillpath": ["distillpath_is16", "distillpath_ks16"]}
FAM = {e: f for f, es in GROUPS.items() for e in es}
ENC = sorted(S)
for e in ENC: FAM.setdefault(e, e)
par = {e: P[e]["params"] / 1e6 for e in ENC}
sh = {e: S[e]["blk_k20_sel"]["share"] for e in ENC}

def _rank(a):
    a = np.asarray(a, float); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(len(a))
    d = collections.defaultdict(list)
    for i, x in enumerate(a): d[x].append(i)
    for x, ix in d.items():
        if len(ix) > 1:
            m = float(np.mean([r[i] for i in ix]))
            for i in ix: r[i] = m
    return r

def spearman(x, y):
    rx, ry = _rank(x) , _rank(y); rx = rx - rx.mean(); ry = ry - ry.mean()
    return float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))

def permP(x, y, n=20000, seed=0):
    rho = spearman(x, y); rng = np.random.default_rng(seed); y = np.asarray(y, float)
    hit = sum(1 for _ in range(n) if abs(spearman(x, rng.permutation(y))) >= abs(rho) - 1e-12)
    return rho, hit / n

fam = collections.defaultdict(list)
for e in ENC: fam[FAM[e]].append(e)
# 一个家族里全部成员都是蒸馏学生时，该家族算蒸馏家族
distilled_fams = {f for f, es in fam.items() if all(e in DISTILLED for e in es)}

def famvals(drop=()):
    ks = [f for f in sorted(fam) if f not in drop]
    return ([float(np.median([par[e] for e in fam[f]])) for f in ks],
            [float(np.median([sh[e] for e in fam[f]])) for f in ks], ks)

out = {}
r, p = permP([par[e] for e in ENC], [sh[e] for e in ENC]); out["enc_all"] = dict(n=len(ENC), rho=r, P=p)
nd = [e for e in ENC if e not in DISTILLED]
r, p = permP([par[e] for e in nd], [sh[e] for e in nd]); out["enc_not_distilled"] = dict(n=len(nd), rho=r, P=p)
x, y, ks = famvals(); r, p = permP(x, y); out["fam_all"] = dict(n=len(ks), rho=r, P=p)
x, y, ks = famvals(distilled_fams); r, p = permP(x, y); out["fam_not_distilled"] = dict(n=len(ks), rho=r, P=p)
out["distilled_encoders"] = sorted(DISTILLED); out["distilled_families"] = sorted(distilled_fams)
json.dump(out, open(f"{R}/family_spearman.json", "w"), indent=1)
for k in ("enc_all", "enc_not_distilled", "fam_all", "fam_not_distilled"):
    print("%-18s n=%2d  r_s=%+.4f  P=%.4f" % (k, out[k]["n"], out[k]["rho"], out[k]["P"]))
