# -*- coding: utf-8 -*-
"""分区对照表（附录）：HEST 57 编码器与 Xenium 16 区域。
HEST：results/hest_controls_{enc}.json（hest_controls.py）+ hest_blocks_{enc}.json（论文口径的图像 oracle blk_k20）+ hest_effres_ps_{enc}.json（官方 α ridge 的逐样本 PCC）。
Xenium：results/xen_controls/*.json（xen_controls.py），ridge 与图像 oracle 的 β 轮廓取自 results/blocks_xen_bands_base（同协议，非 ok bin 以 0 填充后再滤波）。"""
import json, glob, os, numpy as np
R = os.environ.get("S4ST_RESULTS", "results")
encs = sorted(f.split("hest_controls_")[1][:-5] for f in glob.glob(f"{R}/hest_controls_*.json") if "summary" not in f)
rows = []; num = {}
for e in encs:
    C = json.load(open(f"{R}/hest_controls_{e}.json"))["samples"]; B = json.load(open(f"{R}/hest_blocks_{e}.json"))["samples"]; P = json.load(open(f"{R}/hest_rsel_ps_{e}.json"))["per_sample_pcc"]  # 统一基线：留一队列选 α 的 ridge（与 Table 2 相同）
    sids = list(C); img = np.array([B[s]["blk_k20"] for s in sids]); img2 = np.array([C[s]["oracle_image"] for s in sids]); co = np.array([C[s]["oracle_coord"] for s in sids])
    ra = np.array([C[s]["oracle_random"] for s in sids]); tr = np.array([C[s]["trainonly_image"] for s in sids]); rg = np.array([P[s] for s in sids])
    TOC = json.load(open(f"{R}/hest_trainonly_coord.json"))["samples"]; toc = np.array([TOC[s]["trainonly_coord"] for s in sids])  # PAT：坐标分区 + 训练侧均值（与编码器无关）
    IND = f"{R}/hest_inductive_{e}.json"; ind = np.array([json.load(open(IND))["samples"][s]["inductive"] for s in sids]) if os.path.exists(IND) else np.full(len(sids), np.nan)  # 归纳式：分区与均值都只用训练样本
    coh = {}
    for i, s in enumerate(sids): coh.setdefault(C[s]["cohort"], []).append(i)
    cm = lambda v: np.array([np.median(v[ix]) for ix in coh.values()])
    rows.append(dict(enc=e, img=np.median(img), coord=np.median(co), rand=np.median(ra), train=np.median(tr), ridge=np.median(rg), dev=float(np.max(np.abs(img - img2))), ind=float(np.median(ind)), coh_ind_gt_ridge=int((cm(ind) > cm(rg)).sum()), ind_over_ridge=float(np.median(cm(ind) / cm(rg))),
                     coh_img_gt_coord=int((cm(img) > cm(co)).sum()), coh_train_gt_ridge=int((cm(tr) > cm(rg)).sum()), coh_coord_gt_ridge=int((cm(co) > cm(rg)).sum()), tocoord=float(np.median(toc)), coh_tocoord_gt_ridge=int((cm(toc) > cm(rg)).sum()),
                     train_over_ridge=float(np.median(cm(tr) / cm(rg))), img_over_ridge=float(np.median(cm(img) / cm(rg))), rand_over_ridge=float(np.median(cm(ra) / cm(rg))), coord_over_img=float(np.median(cm(co) / cm(img))), rand_over_img=float(np.median(cm(ra) / cm(img))), coord_over_ridge=float(np.median(cm(co) / cm(rg)))))
A = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "enc"}
med = lambda k: float(np.median(A[k])); rng = lambda k: (float(A[k].min()), float(A[k].max()))
num["hest"] = {k: dict(median=med(k), min=rng(k)[0], max=rng(k)[1]) for k in A}
num["hest"]["n_enc"] = len(rows); num["hest"]["dev_gt_1e3"] = int((A["dev"] > 1e-3).sum())
# ── Xenium ──
X = {json.load(open(f))["name"]: json.load(open(f)) for f in glob.glob(f"{R}/xen_controls/*.json")}
Bb = {json.load(open(f))["name"]: json.load(open(f)) for f in glob.glob(f"{R}/blocks_xen_bands_base/*.json")}
SPEC = ["Human_Breast_Biomarkers_S1", "Human_Breast_Biomarkers_S2", "Human_Breast_Biomarkers_S3", "Human_Breast_Biomarkers_S4",
        "Xenium_Prime_Cervical", "Xenium_Prime_Ovarian", "Xenium_V1_Human_Kidney", "Xenium_V1_Human_Ovary",
        "Lung", "Xenium_Prime_Breast_Cancer", "Xenium_Prime_Human_Prostate", "Xenium_Prime_Human_Skin", "Xenium_Prime_Human_Lymph_Node"]   # 2026-09-14 新增 5 个标本
def sp(n):
    if "Human_Lung_Cancer_FFPE" in n: return "Lung"   # Xenium v1 与 Prime 5K 为同一供体同一组织块，按一个标本计
    return next(s for s in SPEC if n.startswith(s))
def specmed(d):
    per = {}
    for n, v in d.items(): per.setdefault(sp(n), []).append(v)
    m = [float(np.median(v)) for v in per.values()]; return float(np.median(m)), min(m), max(m), len(m)
xen = {}
if X:
    for key, fn in [("ridge", lambda d: d["ridge"]["pcc"]), ("img", lambda d: d["K20"]["oracle_image"]["pcc"]), ("coord", lambda d: d["K20"]["oracle_coord"]["pcc"]), ("rand", lambda d: d["K20"]["oracle_random_matched"]["pcc"]),
                    ("train_img", lambda d: d["K20"]["trainonly_image"]["pcc"]), ("train_coord", lambda d: d["K20"]["trainonly_coord"]["pcc"]), ("img200", lambda d: d["K200"]["oracle_image"]["pcc"]), ("coord200", lambda d: d["K200"]["oracle_coord"]["pcc"]), ("train_img200", lambda d: d["K200"]["trainonly_image"]["pcc"])]:
        xen[key] = specmed({n: fn(d) for n, d in X.items()})
    FR = {"img": lambda d: d["K20"]["oracle_image"]["pcc"], "coord": lambda d: d["K20"]["oracle_coord"]["pcc"], "rand": lambda d: d["K20"]["oracle_random_matched"]["pcc"], "train_img": lambda d: d["K20"]["trainonly_image"]["pcc"], "train_coord": lambda d: d["K20"]["trainonly_coord"]["pcc"], "img200": lambda d: d["K200"]["oracle_image"]["pcc"], "coord200": lambda d: d["K200"]["oracle_coord"]["pcc"], "train_img200": lambda d: d["K200"]["trainonly_image"]["pcc"]}
    for k_, f_ in FR.items(): xen[k_ + "_frac"] = specmed({n: f_(d) / d["ridge"]["pcc"] for n, d in X.items()})
    XI = {json.load(open(f))["name"]: json.load(open(f)) for f in glob.glob(f"{R}/xen_inductive/*.json")}
    if len(XI) == len(X): xen["ind_img"] = specmed({n: XI[n]["inductive_partition"]["pcc"] for n in XI}); xen["ind_img_frac"] = specmed({n: XI[n]["inductive_partition"]["pcc"] / XI[n]["ridge"]["pcc"] for n in XI})
    for key, fn in [("coord_over_img", lambda d: d["K20"]["oracle_coord"]["pcc"] / d["K20"]["oracle_image"]["pcc"]), ("train_over_img", lambda d: d["K20"]["trainonly_image"]["pcc"] / d["K20"]["oracle_image"]["pcc"]), ("train_over_ridge", lambda d: d["K20"]["trainonly_image"]["pcc"] / d["ridge"]["pcc"]), ("img_over_ridge", lambda d: d["K20"]["oracle_image"]["pcc"] / d["ridge"]["pcc"])]:
        xen[key] = specmed({n: fn(d) for n, d in X.items()})
    # 退化：同标量 PCC 下的 β1；ridge/oracle 的 β1 取 base 流水线
    deg = {}
    for nm in ["blur", "white_noise", "boundary_shift", "hotspot_removal"]:
        deg[nm] = dict(beta1=specmed({n: d["degradations"][nm]["bands"]["1"] for n, d in X.items()}), pcc=specmed({n: d["degradations"][nm]["pcc"] for n, d in X.items()}),
                       matched=int(sum(d["degradations"][nm]["flag"] == "matched" for d in X.values())), n=len(X), strength=specmed({n: d["degradations"][nm]["strength"] for n, d in X.items()}))
    deg["ridge"] = dict(beta1=specmed({n: Bb[n]["pred"]["ridge"]["band_pcc"] for n in X if n in Bb}), pcc=specmed({n: Bb[n]["pred"]["ridge"]["pcc"] for n in X if n in Bb}))
    deg["oracle"] = dict(beta1=specmed({n: Bb[n]["pred"]["dom20"]["band_pcc"] for n in X if n in Bb}), pcc=specmed({n: Bb[n]["pred"]["dom20"]["pcc"] for n in X if n in Bb}))
    xen["degradations"] = deg; xen["n_regions"] = len(X)
num["xen"] = xen
json.dump(num, open(f"{R}/controls_numbers.json", "w"), indent=1)

# ── 表：Xenium 对照与退化 ──
if X and xen.get("n_regions", 0) >= 16:   # 2026-09-14：区域数由数据决定（原写死 16）
    def f3(t): return f"{t[0]:.3f} & {t[1]:.3f}--{t[2]:.3f}"
    rr = xen["ridge"][0]
    L = [r"\begin{tabular}{lrrr}", r"\toprule", (r"Predictor on Xenium ($%d$ regions, $%d$ specimens) & PCC & Range over specimens & Fraction of ridge \\" % (xen["n_regions"], xen.get("n_specimens", len(SPEC)))), r"\midrule",
         f"Ridge on frozen features & {f3(xen['ridge'])} & 1.00 \\\\",
         f"Image partition, measured means ($K=20$; domain oracle) & {f3(xen['img'])} & {xen['img_frac'][0]:.2f} \\\\",
         f"Coordinate partition, measured means ($K=20$) & {f3(xen['coord'])} & {xen['coord_frac'][0]:.2f} \\\\",
         f"Random matched-size partition, measured means ($K=20$) & {f3(xen['rand'])} & {xen['rand_frac'][0]:.2f} \\\\",
         f"Image partition, means from training blocks ($K=20$) & {f3(xen['train_img'])} & {xen['train_img_frac'][0]:.2f} \\\\",
         f"Image partition learned on training blocks, means from training blocks ($K=20$) & {f3(xen['ind_img'])} & {xen['ind_img_frac'][0]:.2f} \\\\",
         f"Coordinate partition, means from training blocks ($K=20$) & {f3(xen['train_coord'])} & {xen['train_coord_frac'][0]:.2f} \\\\",
         r"\midrule",
         f"Image partition, measured means ($K=200$) & {f3(xen['img200'])} & {xen['img200_frac'][0]:.2f} \\\\",
         f"Coordinate partition, measured means ($K=200$) & {f3(xen['coord200'])} & {xen['coord200_frac'][0]:.2f} \\\\",
         f"Image partition, means from training blocks ($K=200$) & {f3(xen['train_img200'])} & {xen['train_img200_frac'][0]:.2f} \\\\",
         r"\bottomrule", r"\end{tabular}"]
    open("paper/tab_controls_xen.tex", "w").write("\n".join(L) + "\n")
    dg = xen["degradations"]
    L = [r"\begin{tabular}{lrrr}", r"\toprule", r"Field compared with the measurement & Scalar PCC & $\beta_1$ & $\beta_1$ range over specimens \\", r"\midrule",
         f"Trained model (ridge) & {dg['ridge']['pcc'][0]:.3f} & {dg['ridge']['beta1'][0]:.3f} & {dg['ridge']['beta1'][1]:.3f}--{dg['ridge']['beta1'][2]:.3f} \\\\",
         f"Measurement blurred (diffusion, matched) & {dg['blur']['pcc'][0]:.3f} & {dg['blur']['beta1'][0]:.3f} & {dg['blur']['beta1'][1]:.3f}--{dg['blur']['beta1'][2]:.3f} \\\\",
         f"Measurement shifted (boundary displacement, matched) & {dg['boundary_shift']['pcc'][0]:.3f} & {dg['boundary_shift']['beta1'][0]:.3f} & {dg['boundary_shift']['beta1'][1]:.3f}--{dg['boundary_shift']['beta1'][2]:.3f} \\\\",
         f"Measurement plus white noise (matched) & {dg['white_noise']['pcc'][0]:.3f} & {dg['white_noise']['beta1'][0]:.3f} & {dg['white_noise']['beta1'][1]:.3f}--{dg['white_noise']['beta1'][2]:.3f} \\\\",
         f"Measurement with hotspots removed (strongest setting, not matchable) & {dg['hotspot_removal']['pcc'][0]:.3f} & {dg['hotspot_removal']['beta1'][0]:.3f} & {dg['hotspot_removal']['beta1'][1]:.3f}--{dg['hotspot_removal']['beta1'][2]:.3f} \\\\",
         f"Domain oracle ($K=20$) & {dg['oracle']['pcc'][0]:.3f} & {dg['oracle']['beta1'][0]:.3f} & {dg['oracle']['beta1'][1]:.3f}--{dg['oracle']['beta1'][2]:.3f} \\\\",
         r"\bottomrule", r"\end{tabular}"]
    open("paper/tab_degrade.tex", "w").write("\n".join(L) + "\n")
    print("已写 tab_controls_xen.tex, tab_degrade.tex；退化匹配成功数:", {k: v.get("matched") for k, v in dg.items() if "matched" in v}, "| 强度中位:", {k: round(v["strength"][0], 3) for k, v in dg.items() if "strength" in v})
# ── 表：HEST 对照 ──
L = [r"\begin{tabular}{lrrr}", r"\toprule", r"Predictor on the benchmark & PCC (median over encoders) & Range over encoders & Cohorts above ridge \\", r"\midrule"]
def frow(lab, k, coh=None): return f"{lab} & {med(k):.3f} & {rng(k)[0]:.3f}--{rng(k)[1]:.3f} & " + (f"{np.median(A[coh]):.0f}/10" if coh else "--") + r" \\"
L += [frow("Ridge on frozen features (cohort-selected $\\lambda$)", "ridge"), frow("Image partition, measured means (domain oracle)", "img", None), frow("Coordinate partition, measured means", "coord", "coh_coord_gt_ridge"),
      frow("Random matched-size partition, measured means", "rand"), frow("Image partition, means from training samples", "train", "coh_train_gt_ridge"), frow("Coordinate partition, means from training samples", "tocoord", "coh_tocoord_gt_ridge"), frow("Image partition learned on training samples, means from training samples", "ind", "coh_ind_gt_ridge"), r"\bottomrule", r"\end{tabular}"]
open("paper/tab_controls_hest.tex", "w").write("\n".join(L) + "\n")
print(json.dumps({"hest": {k: (round(v["median"], 3), round(v["min"], 3), round(v["max"], 3)) if isinstance(v, dict) else v for k, v in num["hest"].items()}}, indent=0))
if X: print(json.dumps({k: ([round(x, 3) if isinstance(x, float) else x for x in v] if isinstance(v, (list, tuple)) else v) for k, v in xen.items() if k != "degradations"}, indent=0)); print({k: (round(v["beta1"][0], 3), round(v["pcc"][0], 3), v.get("matched")) for k, v in xen["degradations"].items()})
