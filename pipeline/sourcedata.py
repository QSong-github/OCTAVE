# -*- coding: utf-8 -*-
"""抽出每张图实际绘制的逐单元数值，供 source-data 工作簿使用。不重算结论。"""
import glob, json, os, re
import numpy as np
R = "/path/to/systema4ST/results"
SH = {}
def J(p):
    try: return json.load(open(p))
    except Exception: return None
def spec(n):
    m = re.match(r"(Human_Breast_Biomarkers_S\d)_(Top|Mid|Bot)$", n)
    return m.group(1) if m else n
def add(k, cols, rows, src, note=""):
    SH[k] = dict(columns=cols, rows=rows, source=src, note=note)
    print(f"{k:22s} {len(rows):5d} rows x {len(cols)} cols   <- {src}")

# ── 附录图 S1：Moran / 天花板（逐区域）
C = ["region","specimen","pcc","eq_sigma_um","eq_flag","ceiling","c_full","skill",
     "moran_true","moran_pred","moran_diff","pcc_moran_r","pcc_moran_rho","n_bins","n_genes","counts_median"]
rows = []
for f in sorted(glob.glob(f"{R}/xenium/*.json")):
    d = J(f)
    if d: rows.append([d.get("name"), spec(d.get("name",""))] + [d.get(c) for c in C[2:]])
add("FigS1_moran_region", C, rows, "results/xenium/*.json", "每点一个 Xenium 区域")

# ── 附录图 S1：逐基因（PCC 与 Moran I 的基因层级关系）
f0 = sorted(glob.glob(f"{R}/per_gene_xen/*.json"))
if f0:
    MK = ["pcc", "eq", "moran", "moran_pred", "ceil", "mean", "zero"]
    rows = []
    for f in f0:
        d = J(f); n = d["name"]
        for g, v in d["genes"].items():
            rows.append([n, spec(n), g] + [v.get(k) for k in MK])
    add("FigS1_moran_gene", ["region", "specimen", "gene", "pcc", "eq_sigma_um",
        "moran_measured", "moran_predicted", "ceiling", "mean_expr", "zero_frac"],
        rows, "results/per_gene_xen/*.json", "每行一个基因x区域")

# ── 附录图 S3：下游后果（逐区域 × 输出栅格）
rows, cols = [], None
for f in sorted(glob.glob(f"{R}/downstream_*.json")):
    d = J(f)
    if not d: continue
    for b, v in sorted(d["scales"].items(), key=lambda z: int(z[0])):
        if cols is None: cols = sorted(v.keys()); print("  downstream 指标:", cols)
        rows.append([d["name"], spec(d["name"]), int(b), d.get("truth_corr_len_um"), d.get("n_genes")] + [v.get(c) for c in cols])
add("FigS3_downstream", ["region","specimen","pred_bin_um","truth_corr_len_um","n_genes"] + (cols or []),
    rows, "results/downstream_*.json", "同一批预测在四个输出栅格上的六个下游读数")

# ── 正文图 4：评测栅格 跟随 vs 固定
rows = []
for f in sorted(glob.glob(f"{R}/downstream_*.json")):
    n = os.path.basename(f)[len("downstream_"):-5]; d = J(f)
    if not d: continue
    for b in [8, 16, 32, 64]:
        q = J(f"{R}/xenium_multi/{n}_bin{b}.json") if b != 16 else J(f"{R}/xenium/{n}.json")
        fx = d["scales"].get(str(b), {}).get("pcc")
        if q and fx is not None:
            rows.append([n, spec(n), b, q.get("pcc"), fx, q.get("eq_sigma")])
add("Fig4_grid", ["region","specimen","pred_bin_um","pcc_grid_follows","pcc_grid_fixed16","eq_sigma_follows_um"],
    rows, "results/xenium_multi/*, results/xenium/*, results/downstream_*", "同一批预测，两种评测栅格")

# ── 附录图 S2：未报告的协议旋钮
rows = []
for f in sorted(glob.glob(f"{R}/tower_sweep/*.json")):
    s = (J(f) or {}).get("_summary")
    if s: rows.append(["encoder", os.path.basename(f)[:-5], s.get("pcc"), s.get("eq_sigma"), s.get("dim"), s.get("moran_ratio")])
for f in sorted(glob.glob(f"{R}/panel_sweep_hvg*/hibou_l.json")):
    s = (J(f) or {}).get("_summary")
    if s: rows.append(["gene panel", re.search(r"hvg(\d+)", f).group(0), s.get("pcc"), s.get("eq_sigma"), s.get("dim"), s.get("moran_ratio")])
for f in sorted(glob.glob(f"{R}/ctx_sweep/*.json")):
    s = (J(f) or {}).get("_summary")
    if s: rows.append(["context/crop", os.path.basename(f)[:-5], s.get("pcc"), s.get("eq_sigma"), s.get("dim"), s.get("moran_ratio")])
B = J(f"{R}/buffer_cv.json") or {}
for sec, v in B.items():
    for kk, vv in v.items():
        rows.append(["CV leakage buffer", f"{sec} {kk}", (vv.get("pcc") if isinstance(vv, dict) else vv), None, None, None])
add("FigS2_knobs", ["knob","setting","pcc","eq_sigma_um","dim","moran_ratio"],
    rows, "results/tower_sweep, panel_sweep_hvg*, ctx_sweep, buffer_cv.json", "每个旋钮作用在固定预测上")

# ── 附录图 S2：归一化目标（log1p vs cp10k）
enc = sorted(os.path.basename(f)[len("hest_cp10k_"):-5] for f in glob.glob(f"{R}/hest_cp10k_*.json"))
rows = []
for e in enc:
    a, b = J(f"{R}/hest_reported_pcc_{e}.json"), J(f"{R}/hest_cp10k_{e}.json")
    if not a or not b: continue
    pa = a.get("per_sample_pcc", a); pb = b.get("per_sample_pcc", b)
    if isinstance(pa, dict) and isinstance(pb, dict):
        for s in sorted(set(pa) & set(pb)): rows.append([e, s, pa[s], pb[s]])
add("FigS2_target", ["encoder","sample","pcc_log1p","pcc_cp10k"],
    rows, "results/hest_reported_pcc_*.json, results/hest_cp10k_*.json", "同一管线只换表达目标")

json.dump(SH, open(f"{R}/sourcedata.json", "w"))
print("\n共", len(SH), "张表 ->", f"{R}/sourcedata.json")
