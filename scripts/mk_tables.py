# -*- coding: utf-8 -*-
"""由冻结的 JSON 生成正文与附录的 LaTeX 表格。不手抄数字。"""
import json, glob, os, re
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R, OUT = os.path.join(BASE, "results"), os.path.join(BASE, "paper")
W = lambda n, L: open(os.path.join(OUT, n), "w").write("\n".join(L) + "\n")

def spec(n):
    m = re.match(r"(Human_Breast_Biomarkers_S\d)_(Top|Mid|Bot)$", n)
    return m.group(1) if m else n

SHORT = {"Human_Breast_Biomarkers_S1": "Breast S1", "Human_Breast_Biomarkers_S2": "Breast S2",
         "Human_Breast_Biomarkers_S3": "Breast S3", "Human_Breast_Biomarkers_S4": "Breast S4",
         "Xenium_Prime_Cervical_Cancer_FFPE": "Cervical",
         "Xenium_Prime_Ovarian_Cancer_FFPE_XRrun": "Ovarian (Prime)",
         "Xenium_V1_Human_Kidney_FFPE_Protein_updated": "Kidney",
         "Xenium_V1_Human_Ovary_Cancer_FF": "Ovary (V1)"}
NAME = {"dinov3_vitl16": "DINOv3-L", "dinov2_large": "DINOv2-L", "kaiko_vitb16": "Kaiko-B16",
        "kaiko_vitl14": "Kaiko-L14", "kaiko_vits16": "Kaiko-S16", "lunit_vits8": "Lunit-S8",
        "conch_v15": "CONCH v1.5", "hoptimus0": "H-optimus-0", "midnight12k": "Midnight-12k",
        "phikon_v2": "Phikon-v2", "phikon": "Phikon", "uni_v2": "UNI v2",
        "virchow2": "Virchow2", "gigapath": "Prov-GigaPath", "ciga": "CIGA",
        # 2026-08-29 扩集
        "hoptimus1": "H-optimus-1", "h0_mini": "H0-mini", "genbio_pathfm": "GenBio-PathFM",
        "omiclip": "OmiCLIP", "plip": "PLIP", "quiltnet": "QuiltNet", "keep": "KEEP",
        "openmidnight": "OpenMidnight", "hibou_l": "Hibou-L", "uni_v1": "UNI",
        "virchow": "Virchow", "conch_v1": "CONCH", "musk": "MUSK",
        "phaet": "Phaet", "mascaret": "Mascaret"}
# 每个被评测模型的出处。H-optimus-0 无论文，按模型发布引；UNI v2 无独立论文，
# 沿用原 UNI（同作者后续版本）；CONCH v1.5 随 TITAN 一同发布。
CITE = {"dinov3_vitl16": "dinov3", "dinov2_large": "dinov2", "kaiko_vitb16": "kaiko",
        "kaiko_vitl14": "kaiko", "kaiko_vits16": "kaiko", "lunit_vits8": "lunit",
        "conch_v15": "titan", "hoptimus0": "hoptimus", "midnight12k": "midnight",
        "phikon_v2": "phikonv2", "phikon": "phikon", "uni_v2": "uni",
        "virchow2": "virchow2", "gigapath": "gigapath", "ciga": "ciga",
        "hoptimus1": "hoptimus1", "h0_mini": "h0mini", "genbio_pathfm": "genbiopathfm",
        "omiclip": "omiclip", "plip": "plip", "quiltnet": "quiltnet", "keep": "keep",
        "openmidnight": "openmidnight", "hibou_l": "hibou", "uni_v1": "uni",
        "virchow": "virchow", "conch_v1": "conch", "musk": "musk",
        "phaet": "waivrobust", "mascaret": "waivrobust"}
# 2026-09-03 扩集到 49 个编码器：新 19 个的显示名与引用键
NAME.update({"kaiko_vitb8": "Kaiko-B8", "lunit_vits16": "Lunit-S16", "lunit_r50_swav": "Lunit-R50-SwAV",
             "lunit_r50_bt": "Lunit-R50-BT", "lunit_r50_moco": "Lunit-R50-MoCo", "ctranspath": "CTransPath",
             "gpfm": "GPFM", "retccl": "RetCCL", "dinov2_base": "DINOv2-B", "dinov2_giant": "DINOv2-g",
             "dinov3_vitb16": "DINOv3-B", "dinov3_vith16": "DINOv3-H+", "clip_vitl14": "CLIP-L/14",
             "pathgen_clip": "PathGen-CLIP", "siglip2": "SigLIP 2", "biomedclip": "BiomedCLIP",
             "hibou_b": "Hibou-B", "gigapath_flash": "GigaPath-Flash", "path_foundation": "Path Foundation", "mstar": "mSTAR", "distillpath_ks16": "DistillPath-KS16", "distillpath_is16": "DistillPath-IS16", "litefm": "LiteFM", "litefm_s": "LiteFM-S", "litefm_l": "LiteFM-L", "litevirchow2": "LiteVirchow2", "pathryoshka_b": "Pathryoshka-B"})
CITE.update({"kaiko_vitb8": "kaiko", "lunit_vits16": "lunit", "lunit_r50_swav": "lunit", "lunit_r50_bt": "lunit",
             "lunit_r50_moco": "lunit", "ctranspath": "ctranspath", "gpfm": "gpfm", "retccl": "retccl",
             "dinov2_base": "dinov2", "dinov2_giant": "dinov2", "dinov3_vitb16": "dinov3", "dinov3_vith16": "dinov3",
             "clip_vitl14": "clip", "pathgen_clip": "pathgen", "siglip2": "siglip2", "biomedclip": "biomedclip",
             "hibou_b": "hibou", "gigapath_flash": "gigapath", "path_foundation": "pathfoundation", "mstar": "mstar", "distillpath_ks16": "distillpath", "distillpath_is16": "distillpath", "litefm": "litepath", "litefm_s": "litepath", "litefm_l": "litepath", "litevirchow2": "litepath", "pathryoshka_b": "pathryoshka"})
LBL = lambda e: "%s \\citep{%s}" % (NAME.get(e, e), CITE[e]) if e in CITE else NAME.get(e, e)

# ═══ 表 1（正文）逐样本：标量落差 vs 最细带落差
S = {}
for f in sorted(glob.glob(os.path.join(R, "blocks_xen_bands", "*.json"))):
    d = json.load(open(f)); P = d["pred"]
    ts = sorted(int(t) for t in P["ridge"]["bands"])
    ov = 100 * (P["ridge"]["pcc"] - P["dom20"]["pcc"]) / P["ridge"]["pcc"]
    r, b = P["ridge"]["bands"][str(ts[0])], P["dom20"]["bands"][str(ts[0])]
    S.setdefault(spec(d["name"]), []).append((ov, 100 * (r - b) / r, P["ridge"]["pcc"], P["dom20"]["pcc"]))
rows = []
for k, v in S.items():
    a = np.array(v)
    # 比值：**先逐区域算比值**再在样本内取中位 —— 与正文口径一致。
    # 若改成「样本内中位数相除」，多区域样本会给出不同的数（3.36 vs 3.61）。
    rows.append((SHORT.get(k, k), len(v), np.median(a[:, 2]), np.median(a[:, 3]),
                 np.median(a[:, 0]), np.median(a[:, 1]), float(np.median(a[:, 1] / a[:, 0]))))
rows.sort(key=lambda z: -z[6])
t = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
     "Specimen & Regions & $\\mathrm{PCC}_{\\mathrm{mod}}$ & $\\mathrm{PCC}_{\\mathrm{dom}}$ "
     "& $\\Delta_{\\mathrm{scalar}}$ & $\\Delta_{\\mathrm{fine}}$ & Ratio \\\\", "\\midrule"]
for n, k, pm, pb, o, f_, rt in rows:
    t.append(f"{n} & {k} & {pm:.3f} & {pb:.3f} & {o:.1f}\\% & {f_:.1f}\\% & {rt:.2f} \\\\")
t += ["\\midrule", f"Median & & & & {np.median([z[4] for z in rows]):.1f}\\% & "
      f"{np.median([z[5] for z in rows]):.1f}\\% & \\textbf{{{np.median([z[6] for z in rows]):.2f}}} \\\\"]
sp = os.path.join(R, "seed_spread.json")
if os.path.exists(sp):
    S = json.load(open(sp))["summary"]
    # seed 0 是冻结配置，其逐样本值即上表各行；这一行给 16 个种子重跑整条链的散布。
    assert abs(S["headline_ratio_seed0"] - np.median([z[6] for z in rows])) < 5e-3
    t.append(f"{S['n_seeds']} seeds & & & & ${S['scalar_median']:.1f} \\pm {S['scalar_sd']:.1f}\\%$ & "
             f"${S['fine_median']:.1f} \\pm {S['fine_sd']:.1f}\\%$ & "
             f"${S['headline_ratio_median']:.2f}\\,[{S['headline_ratio_min']:.2f}, {S['headline_ratio_max']:.2f}]$ \\\\")
    print(f"        种子行: 比值 {S['headline_ratio_median']:.2f} [{S['headline_ratio_min']:.2f}, {S['headline_ratio_max']:.2f}] "
          f"sd {S['headline_ratio_sd']:.3f}; {S['seeds_all_specimens_agree']}/{S['n_seeds']} 个种子给 8/8; "
          f"ARI 中位 {S['ari_median']:.3f} [{S['ari_min']:.3f}, {S['ari_max']:.3f}]")
t += ["\\bottomrule", "\\end{tabular}"]
W("tab_specimens.tex", t)
print(f"表 1 逐样本: {len(rows)} 样本, 比值中位 {np.median([z[6] for z in rows]):.2f}")

# ═══ 表 2（正文）benchmark：排行榜、各自的匹配检索地板、跨队列误差棒
K = json.load(open(os.path.join(R, "k_sensitivity.json")))
KS = sorted(int(x) for x in K)
CS = json.load(open(os.path.join(R, "cohort_spread.json")))
mod = {e: K["50"][e]["floor"] + K["50"][e]["gap"] for e in K["50"]}
ENC = sorted(mod, key=lambda e: -mod[e])
REL = lambda e: (K["50"][e]["rel_ok"] if K["50"][e]["rel_ok"] is not None else K["50"][e]["rel"])
best_m, best_r = max(mod.values()), max(REL(e) for e in ENC)
best_w = max(K["50"][e]["n_cohorts_win"] for e in ENC)
# 报告分与地板用基准自己的逐样本加权；误差棒在队列层级，即本文的推断单元。
# 两者不是同一个加权，图注里必须写明，不许当成同一个数。
for e in ENC:
    assert abs(CS[e]["floor_mean_sample"] - K["50"][e]["floor"]) < 2e-3, e
    assert CS[e]["cohorts_won"] == K["50"][e]["n_cohorts_win"], e   # 口径必须同源
# 27 行单栏会把正文顶到第 10 页；排成左右两栏，高度减半，内容不变。
HDR = ("Encoder & PCC & Floor & Margin (\\%) & Margin per cohort & Won")
t = ["\\begin{tabular}{lrrrr r@{\\hskip 12pt} lrrrr r}", "\\toprule",
     HDR + " & " + HDR + " \\\\", "\\midrule"]
def row(e):
    r, o = K["50"][e], CS[e]
    m = f"\\textbf{{{mod[e]:.4f}}}" if mod[e] == best_m else f"{mod[e]:.4f}"
    rs = f"$\\mathbf{{{REL(e):+.1f}}}$" if REL(e) == best_r else f"${REL(e):+.1f}$"
    ws = f"\\textbf{{{r['n_cohorts_win']}/10}}" if r["n_cohorts_win"] == best_w else f"{r['n_cohorts_win']}/10"
    return (f"{LBL(e)} & {m} & {r['floor']:.4f} & {rs} & "
            f"${o['cohort_mean']:+.3f}\\pm{o['cohort_sem']:.3f}$ & {ws}")
h = (len(ENC) + 1) // 2
for i in range(h):
    L = row(ENC[i])
    Rt = row(ENC[i + h]) if i + h < len(ENC) else " & " * 5
    t.append(L + " & " + Rt + " \\\\")
lo = sum(1 for e in ENC if K["50"][e]["gap"] < 0)
tmax = max(abs(CS[e]["t"]) for e in ENC)
t += ["\\bottomrule", "\\end{tabular}"]
W("tab_bench.tex", t)
flip = [e for e in ENC if (CS[e]["margin_mean_sample"] > 0) != (CS[e]["cohort_mean"] > 0)]
print(f"表 2 benchmark: {len(ENC)} 编码器, {lo} 个低于自身地板, 榜首 {NAME[ENC[0]]} {mod[ENC[0]]:.4f}")
print(f"        跨队列 |t| 最大 {tmax:.2f}（无一到 2）；换加权符号翻转 {len(flip)}/{len(ENC)}: {flip}")

# ═══ 表 3（正文）k 扫描汇总
t = ["\\begin{tabular}{rrrrrr}", "\\toprule",
     "$k$ & Encoders & Median margin (\\%) & Range (\\%) & Below own floor & Significant \\\\", "\\midrule"]
for k in KS:
    d = K[str(k)]
    v = np.array([(r["rel_ok"] if r["rel_ok"] is not None else r["rel"]) for r in d.values()])
    nb = sum(1 for r in d.values() if (r["rel_ok"] if r["rel_ok"] is not None else r["rel"]) < 0)
    ns = sum(1 for r in d.values() if (r["P_ok"] if r.get("P_ok") is not None else r["P"]) < 0.05)
    t.append(f"{k} & {len(d)} & {np.median(v):+.1f} & {v.min():+.1f} to {v.max():+.1f} & "
             f"{nb}/{len(d)} & {ns}/{len(d)} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
W("tab_ksweep.tex", t)
print("表 3 k 扫描: " + ", ".join(f"k={k}:{sum(1 for r in K[str(k)].values() if (r['rel_ok'] if r['rel_ok'] is not None else r['rel'])<0)}/{len(K[str(k)])} 低于地板" for k in KS))

# ═══ 表 4（正文）评测栅格
G = json.load(open(os.path.join(R, "grid_reversal.json")))["grid_reversal"]
t = ["\\begin{tabular}{lrrrr}", "\\toprule",
     "Evaluation grid & Median (\\%) & Range (\\%) & Regions $>0$ & Specimens $>0$ \\\\", "\\midrule",
     f"Follows the prediction & \\textbf{{{G['follow_median']:+.1f}}} & {G['follow_lo']:+.1f} to {G['follow_hi']:+.1f} & "
     f"{G['n_regions']}/{G['n_regions']} & {G['n_specimens']}/{G['n_specimens']} \\\\",
     f"Fixed at $16\\,\\mu$m & \\textbf{{{G['fixed_median']:+.1f}}} & {G['fixed_lo']:+.1f} to {G['fixed_hi']:+.1f} & "
     f"0/{G['n_regions']} & 0/{G['n_specimens']} \\\\", "\\midrule",
     f"Difference (pp) & {G['swing_pp']:.1f}\\,pp & & {G['n_regions_reverse']}/{G['n_regions']} reverse & "
     f"{G['n_specimens_reverse']}/{G['n_specimens']} reverse \\\\", "\\bottomrule", "\\end{tabular}"]
W("tab_grid.tex", t)
print(f"表 4 栅格: 摆动 {G['swing_pp']:.1f} pp, 反转 {G['n_specimens_reverse']}/{G['n_specimens']} 样本, P={G['P_specimen']:.4f}")

# ═══ 表 5（附录）逐编码器 × k
t = ["\\begin{tabular}{l" + "r" * len(KS) + "r}", "\\toprule",
     "Encoder & " + " & ".join(f"$k={k}$" for k in KS) + " & Cohorts won ($k=50$) \\\\", "\\midrule"]
for e in sorted(K["50"], key=lambda x: -(K["50"][x]["rel_ok"] or K["50"][x]["rel"])):
    c = []
    for k in KS:
        r = K[str(k)].get(e)
        c.append("---" if r is None else f"{(r['rel_ok'] if r['rel_ok'] is not None else r['rel']):+.1f}")
    t.append(f"{LBL(e)} & " + " & ".join(c) + f" & {K['50'][e]['n_cohorts_win']}/10 \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
W("tab_encoders.tex", t)
print(f"表 5 附录 逐编码器 × k: {len(K['50'])} × {len(KS)}")

# ═══ 表 6（附录）超参消融：岭回归 alpha、基因数、划分块数 K
ap = os.path.join(R, "ablate_spread.json")
if os.path.exists(ap):
    A = json.load(open(ap))
    ABLBL = {"alpha": ("Ridge $\\lambda$", lambda k: "$10^{%d}$" % round(np.log10(float(k)))),
           "ngene": ("Genes", lambda k: k),
           "K": ("Clusters $K$", lambda k: k)}
    t = ["\\begin{tabular}{llrrrrrr}", "\\toprule",
         "Axis & Setting & Oracle / model & $\\Delta_{\\mathrm{scalar}}$ & $\\Delta_{\\mathrm{fine}}$ "
         "& Ratio & Specimens & $P$ \\\\", "\\midrule"]
    for ax in ("alpha", "ngene", "K"):
        if ax not in A: continue
        name, fmt = ABLBL[ax]
        keys = sorted(A[ax]["levels"], key=float)
        for i, k in enumerate(keys):
            o = A[ax]["levels"][k]
            d = (k == A[ax]["default"])
            lv = fmt(k)
            if d:
                # 数学模式里 \textbf 不生效，alpha 档用 \mathbf，其余是纯文本
                lv = (lv.replace("$10^", "$\\mathbf{10^").replace("}$", "}}$")
                      if ax == "alpha" else f"\\textbf{{{lv}}}")
            cells = [lv, f"{o['share']:.1f}\\%", f"{o['scalar']:.1f}\\%", f"{o['fine']:.1f}\\%",
                     f"{o['ratio']:.2f}", f"{o['specimens_agree']}/{o['n_specimens']}", f"{o['P']:.4f}"]
            if d: cells = [cells[0]] + [f"\\textbf{{{c}}}" for c in cells[1:]]
            t.append((name if i == 0 else "") + " & " + " & ".join(cells) + " \\\\")
        if ax != "K": t.append("\\midrule")
    t += ["\\bottomrule", "\\end{tabular}"]
    W("tab_ablate.tex", t)
    n = sum(len(A[ax]["levels"]) for ax in A)
    allw = all(v["specimens_agree"] == v["n_specimens"] for ax in A for v in A[ax]["levels"].values())
    rr = [v["ratio"] for ax in A for v in A[ax]["levels"].values()]
    print(f"表 6 消融: {n} 个档位, 比值跨越 {min(rr):.2f}–{max(rr):.2f}, "
          + ("每一档都 8/8 同向" if allw else "**有档位不是全同向**"))
else:
    print("表 6 消融: ablate_spread.json 尚未生成，跳过")

# ═══ 附录：PCA 前各维标准化的稳健性（OpenMidnight 例外的反事实），K=20
zp = os.path.join(R, "hest_blocks_z_summary.json")
if os.path.exists(zp):
    ZS = json.load(open(zp))
    ZS = sorted(ZS, key=lambda r: r["k20"]["orc_z"] - r["k20"]["orc"])
    t = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
         "Encoder & Oracle & Oracle, std. & Difference & Ratio & Ratio, std. & Cohorts \\\\", "\\midrule"]
    for r in ZS:
        v = r["k20"]
        t.append(f"{LBL(r['enc'])} & {v['orc']:.3f} & {v['orc_z']:.3f} & {v['orc_z']-v['orc']:+.3f} & {v['share']:.0f}\\% & {v['share_z']:.0f}\\% & {v['std'][2]}/{v['zs'][2]} \\\\")
    d = np.array([abs(r["k20"]["orc_z"] - r["k20"]["orc"]) for r in ZS if r["enc"] != "openmidnight"])
    ds = np.array([abs(r["k20"]["share_z"] - r["k20"]["share"]) for r in ZS if r["enc"] != "openmidnight"])
    shz = np.array([r["k20"]["share_z"] for r in ZS]); sh0 = np.array([r["k20"]["share"] for r in ZS])
    o0 = np.array([r["k20"]["orc"] for r in ZS]); oz = np.array([r["k20"]["orc_z"] for r in ZS])
    t += ["\\midrule", f"\\textbf{{Median of {len(ZS)}}} & {np.median(o0):.3f} & {np.median(oz):.3f} & & {np.median(sh0):.0f}\\% & {np.median(shz):.0f}\\% & \\\\",
          "\\bottomrule", "\\end{tabular}"]
    W("tab_zscore.tex", t)
    print(f"表 z-score 稳健性: {len(ZS)} 个；除 OpenMidnight 外预言机分变化最大 {d.max():.3f}，占比变化最大 {ds.max():.1f} 点；z-score 后占比中位 {np.median(shz):.0f}% ({shz.min():.0f}–{shz.max():.0f})")

# ═══ 表 7（附录）五个已发表方法 vs 编码器与其匹配检索地板
mp = os.path.join(R, "methods_vs_floor.json")
if os.path.exists(mp):
    MV = json.load(open(mp))
    ref = MV["_reference"]
    rows = sorted(((k, v) for k, v in MV.items() if k != "_reference"),
                  key=lambda kv: -kv[1]["mean_sample"])
    t = ["\\begin{tabular}{lrrrr}", "\\toprule",
         "Method & PCC & Below encoders & Margin per cohort & Cohorts won \\\\",
         "\\midrule"]
    MCITE = {"HisToGene": "histogene", "Hist2ST": "hist2st", "BLEEP": "bleep",
             "HECLIP": "heclip", "HGGEP": "hggep", "THItoGene": "thitogene"}
    for k, v in rows:
        nm = "%s \\citep{%s}" % (k, MCITE[k]) if k in MCITE else k
        t.append(f"{nm} & {v['mean_sample']:.4f} & {v['n_below_encoders']}/{v['n_encoders']} & "
                 f"${v['cohort_mean']:+.4f} \\pm {v['cohort_sem']:.4f}$ & "
                 f"{v['cohorts_won']}/{v['n_cohorts']} \\\\")
    em, fm = ref["encoder_mean"], ref["floor_mean"]
    t += ["\\midrule",
          f"Frozen encoder and ridge & {min(em.values()):.4f}--{max(em.values()):.4f} & & & \\\\",
          "\\bottomrule", "\\end{tabular}"]
    W("tab_methods.tex", t)
    allb = all(v["n_below_floors"] == v["n_floors"] for _, v in rows)
    print(f"表 7 已发表方法: {len(rows)} 个, 全部低于全部地板={allb}, "
          f"最高 {rows[0][0]} {rows[0][1]['mean_sample']:.4f} vs 最低地板 {min(fm.values()):.4f}")
else:
    print("表 7: methods_vs_floor.json 未找到，跳过")

# ═══ 表 8（附录）替代回归头：同一份 PCA-256 特征上，MLP（L2 扫描 / 小隐藏层 / 不早停）、
#     恒等激活 MLP（优化对照）、RBF 核岭回归，对三个编码器在每个队列 fold 0 上与岭回归配对比较。
#     mlp_groupval 留出整片做早停但因此少用一片训练数据，有混杂，不进表（保留在冻结 JSON 里）。
import glob as _gg
sw = {}
for f in sorted(_gg.glob(os.path.join(R, "mlp_head_sweep*.json"))):
    for e, d in json.load(open(f)).items():
        for c, v in d.items():
            sw.setdefault(e, {}).setdefault(c, {"n_train_sections": v["n_train_sections"], "variants": {}})["variants"].update(v["variants"])
if sw:
    HN = [("ridge", "Ridge, $\\alpha = 100/(DG)$ (benchmark protocol)"),
          ("ridge_a100", "Ridge, $\\alpha=10^{2}$"), ("ridge_a1e3", "Ridge, $\\alpha=10^{3}$"),
          ("ridge_a1e4", "Ridge, $\\alpha=10^{4}$"), ("ridge_a1e5", "Ridge, $\\alpha=10^{5}$"),
          ("mlp_identity", "MLP, identity activation, $\\lambda=10^{-4}$"),
          ("mlp_a1e-4", "MLP, $\\lambda=10^{-4}$"), ("mlp_a1", "MLP, $\\lambda=1$"), ("mlp_a10", "MLP, $\\lambda=10$"),
          ("mlp_a30", "MLP, $\\lambda=30$"), ("mlp_a100", "MLP, $\\lambda=100$"), ("mlp_a300", "MLP, $\\lambda=300$"),
          ("mlp_a1000", "MLP, $\\lambda=1000$"),
          ("mlp_h64", "MLP, 64 hidden, $\\lambda=10^{-4}$"),
          ("mlp_fixed100", "MLP, $\\lambda=10^{-4}$, no early stopping"), ("krr_rbf", "Kernel ridge, RBF")]
    ES = [e for e in ("uni_v2", "hoptimus1", "ciga") if e in sw]
    def cell(e, v):
        cs = sorted(c for c in sw[e] if v in sw[e][c]["variants"] and "ridge" in sw[e][c]["variants"])
        if not cs: return None
        te = [sw[e][c]["variants"][v]["test_pcc"] for c in cs]; rg = [sw[e][c]["variants"]["ridge"]["test_pcc"] for c in cs]
        tr = [sw[e][c]["variants"][v]["train_pcc"] for c in cs]
        return (sum(te) / len(te), sum(te) / len(te) - sum(rg) / len(rg), sum(1 for a, b in zip(te, rg) if a > b), len(cs), sum(tr) / len(tr))
    t = ["\\begin{tabular}{l" + "rrr" * len(ES) + "}", "\\toprule",
         "Head & " + " & ".join("\\multicolumn{3}{c}{%s}" % NAME.get(e, e) for e in ES) + " \\\\",
         " & ".join([""] + ["PCC & $\\Delta$ & Won"] * len(ES)) + " \\\\", "\\midrule"]
    for key, lab in HN:
        cells = [cell(e, key) for e in ES]
        if all(x is None for x in cells): continue
        row = [lab]
        for x in cells:
            row += ["--", "--", "--"] if x is None else ([f"{x[0]:.4f}", "", ""] if key == "ridge" else [f"{x[0]:.4f}", f"${x[1]:+.3f}$", f"{x[2]}/{x[3]}"])
        t.append(" & ".join(row) + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}"]
    W("tab_heads.tex", t)
    print(f"表 8 替代回归头: {len(ES)} 编码器 × {sum(1 for k,_ in HN if any(cell(e,k) for e in ES))} 头")
    for e in ES:
        r, b, f10 = cell(e, "ridge"), cell(e, "mlp_a1e-4"), cell(e, "mlp_fixed100")
        best = max((k for k, _ in HN if k not in ("ridge", "mlp_identity") and cell(e, k)), key=lambda k: cell(e, k)[1])
        print(f"   {NAME.get(e,e):<12s} 训练PCC ridge {r[4]:.3f} / MLP {b[4]:.3f} / 不早停 {f10[4]:.3f}；"
              f"测试Δ MLP {b[1]:+.3f}，不早停 {f10[1]:+.3f}；最佳非线性头 {best} {cell(e,best)[1]:+.3f}（赢 {cell(e,best)[2]}/{cell(e,best)[3]}）")
        secs = sorted((sw[e][c]["n_train_sections"], c, sw[e][c]["variants"]["mlp_a1e-4"]["test_pcc"] - sw[e][c]["variants"]["ridge"]["test_pcc"]) for c in sw[e] if "mlp_a1e-4" in sw[e][c]["variants"])
        print("      默认 MLP 逐队列 Δ（按训练切片数）: " + "  ".join(f"{c}({n}片){d:+.3f}" for n, c, d in secs))

# ═══ 表 9（附录）替代头在全部 30 个编码器上：每个标签一张表，与各自 k=50 地板、队列层级误差棒。
#     标签：mlp10 / mlp100 = MLP λ=10 / 100（种子 0 进表，种子 sd 另列）；rsel = 留一队列选 α 的岭回归（无种子）。
# 2026-09-01：检索地板线已撤出论文。以下替代头的表只在设了 S4ST_FLOOR_TABLES=1 时生成。
if os.environ.get("S4ST_FLOOR_TABLES") == "1":
  LABEL = {"mlp10": "MLP, $\\lambda=10$", "mlp100": "MLP, $\\lambda=100$", "rsel": "Ridge, $\\alpha$ selected per cohort"}
  K0 = json.load(open(os.path.join(R, "k_sensitivity.json")))["50"]
  RL = lambda d, e: (d[e]["rel_ok"] if d[e].get("rel_ok") is not None else d[e]["rel"])
  PV = lambda d, e: (d[e].get("P_ok") if d[e].get("P_ok") is not None else d[e]["P"])
  for MT in os.environ.get("MLP_TAGS", "mlp10,mlp100,rsel").split(","):
      kp, cp = os.path.join(R, f"k_sensitivity_{MT}.json"), os.path.join(R, f"cohort_spread_{MT}.json")
      if not (os.path.exists(kp) and os.path.exists(cp)):
          print(f"表 9[{MT}]: 结果尚未到位，跳过"); continue
      KM, CM = json.load(open(kp))["50"], json.load(open(cp))
      EM = sorted(KM, key=lambda e: -(KM[e]["floor"] + KM[e]["gap"]))
      sd = {}
      for e in EM:
          ms = [json.load(open(f))["mean_pcc"] for f in sorted(_gg.glob(os.path.join(R, "mlp_seeds", f"hest_{MT}_ps_{e}_s*.json")))]
          sd[e] = (float(np.std(ms, ddof=1)) if len(ms) > 1 else None)
      has_sd = any(v is not None for v in sd.values())
      t = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
           f"Encoder & Ridge & {LABEL.get(MT, MT)} & Floor & Margin (\\%) & Margin per cohort & Won \\\\", "\\midrule"]
      for e in EM:
          m = KM[e]["floor"] + KM[e]["gap"]; r0 = K0[e]["floor"] + K0[e]["gap"]
          mm = f"{m:.4f}" + (f" $\\pm$ {sd[e]:.4f}" if has_sd and sd[e] is not None else "")
          t.append(f"{LBL(e)} & {r0:.4f} & {mm} & {KM[e]['floor']:.4f} & ${RL(KM, e):+.1f}$ & "
                   f"${CM[e]['cohort_mean']:+.3f}\\pm{CM[e]['cohort_sem']:.3f}$ & {CM[e]['cohorts_won']}/{CM[e]['n_cohorts']} \\\\")
      t += ["\\bottomrule", "\\end{tabular}"]
      W(f"tab_{MT}.tex", t)
      dl = [KM[e]["floor"] + KM[e]["gap"] - (K0[e]["floor"] + K0[e]["gap"]) for e in EM]
      tt = {e: abs(CM[e]["t"]) for e in EM}
      print(f"表 9[{MT}]: {len(EM)} 编码器；头−官方ridge 中位 {np.median(dl):+.4f}（为正 {sum(1 for x in dl if x>0)}/{len(dl)}）；"
            f"低于地板 {sum(1 for e in EM if RL(KM,e)<0)}/{len(EM)}；|t|≥2 {sum(1 for e in EM if tt[e]>=2)}/{len(EM)}"
            f"（余量为正 {sum(1 for e in EM if tt[e]>=2 and CM[e]['cohort_mean']>0)}）；"
            f"队列层级显著 {sum(1 for e in EM if PV(KM,e)<0.05)}/{len(EM)} {[e for e in EM if PV(KM,e)<0.05]}"
            + (f"；种子 sd 中位 {np.median([v for v in sd.values() if v is not None]):.4f}" if has_sd else ""))

# ═══ 表（正文）块预言机在 30 个编码器 × 72 个 HEST 样本上：命题 1 的实验证实。
#     统计量一律 PCC 单位的配对差（队列内中位 → 10 队列均值±sem），占比只在聚合后算一次。
#     分母用留一队列选 α 的 ridge：官方 α=100/(D·G) 欠正则，会压低分母把占比抬高。
bp = os.path.join(R, "hest_blocks_summary.json")
if os.path.exists(bp):
    BS = json.load(open(bp))
    S = lambda r: r["blk_k20_sel"]
    BS.sort(key=lambda r: -S(r)["mod"])          # 按模型分排序，即领域自己的榜单顺序
    PARM = json.load(open(os.path.join(R, "encoder_params.json"))) if os.path.exists(os.path.join(R, "encoder_params.json")) else {}
    def pm(e):
        v = (PARM.get(e) or {}).get("params")
        return "--" if not v else "%.0f" % (v / 1e6)
    t = ["\\begin{tabular}{lrrrrrrr}", "\\toprule",
         "Encoder & Params & Model & Oracle & Ratio & Difference & $|t|$ & Cohorts \\\\", "\\midrule"]
    for r in BS:
        a = S(r)
        t.append(f"{LBL(r['enc'])} & {pm(r['enc'])} & {a['mod']:.4f} & {a['blk']:.4f} & {a['share']:.0f}\\% & "
                 f"${a['cohort_mean']:+.3f}\\pm{a['cohort_sem']:.3f}$ & {abs(a['t']):.1f} & {a['won']}/10 \\\\")
    md = lambda f: float(np.median([f(r) for r in BS]))
    pos = sum(1 for r in BS if S(r)["cohort_mean"] > 0)
    sig = sum(1 for r in BS if S(r)["P"] < 0.05 and S(r)["cohort_mean"] > 0)
    pv = sorted((PARM.get(r["enc"]) or {}).get("params") or 0 for r in BS)
    pv = [x for x in pv if x]
    prange = "%.0f--%.0f" % (min(pv) / 1e6, max(pv) / 1e6) if pv else "--"
    t += ["\\midrule",
          f"\\textbf{{Median of {len(BS)}}} & {prange} & {md(lambda r: S(r)['mod']):.4f} & {md(lambda r: S(r)['blk']):.4f} & "
          f"{md(lambda r: S(r)['share']):.0f}\\% & $\\mathbf{{{md(lambda r: S(r)['cohort_mean']):+.3f}}}$ & "
          f"{md(lambda r: abs(S(r)['t'])):.1f} & \\textbf{{{pos}/{len(BS)}}} \\\\",
          "\\bottomrule", "\\end{tabular}"]
    W("tab_hestblocks.tex", t)
    sh = [S(r)["share"] for r in BS]
    t2 = sum(1 for r in BS if abs(S(r)["t"]) >= 2)
    print(f"表 块预言机×30编码器（单栏，按模型分排序）: 占比中位 {np.median(sh):.0f}% ({min(sh):.0f}–{max(sh):.0f})；"
          f"配对差为正 {pos}/{len(BS)}，|t|≥2 {t2}/{len(BS)}，符号检验 {sig}/{len(BS)}；"
          f"K=50 中位 {np.median([r['blk_k50_sel']['share'] for r in BS]):.0f}%，"
          f"K=200 中位 {np.median([r['blk_k200_sel']['share'] for r in BS]):.0f}%")
