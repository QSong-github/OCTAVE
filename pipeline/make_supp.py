#!/usr/bin/env python
"""补充图 S1–S13。样式与主图共用（统一调色板）。

原则与主图一致：
  · 只从 results/ 读数，不写死数值；
  · 任何跨条件的比值，先查两侧删失率再报（§37 的教训）；
  · 区域层级计数一律标为伪重复，显著性只在样本层级施行；
  · 算不出来的就写「算不出来」，不给一个看起来漂亮的错数。
"""
import glob, json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from math import comb

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({
    "font.size": 7, "axes.linewidth": 0.8, "legend.frameon": False,
    "axes.spines.right": False, "axes.spines.top": False,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.titlesize": 7, "axes.titlepad": 8,
})
P = {"blue": "#0F4D92", "blue2": "#3775BA", "red": "#B64342", "teal": "#42949E",
     "violet": "#9A4D8E", "grey_l": "#CFCECE", "grey_m": "#767676", "grey_d": "#4D4D4D",
     "black": "#272727"}
MM = 1 / 25.4
RES, OUT = "results", "figures_supp"


def specimen(n):
    m = re.match(r"(Human_Breast_Biomarkers_S\d)_(Top|Mid|Bot)$", n)
    return m.group(1) if m else n


def signp(k, n):
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def lab(ax, s, x=-0.16, y=1.10):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="left", va="bottom")


def jl(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUT}/{name}.pdf", flush=True)


# ───────────────────────── S1：估计量定义与代数自查 ─────────────────────────
def s1():
    X = [jl(f) for f in sorted(glob.glob(f"{RES}/xenium/*.json"))]
    X = [d for d in X if d]
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 52 * MM))

    ax = axes[0]
    ce = np.array([d["ceiling"] for d in X]); cf = np.array([d["c_full"] for d in X])
    ax.scatter(ce ** 2, cf, s=16, color=P["blue"], zorder=3)
    lo, hi = min(cf.min(), (ce ** 2).min()) * 0.98, max(cf.max(), (ce ** 2).max()) * 1.02
    ax.plot([lo, hi], [lo, hi], "--", color=P["grey_m"], lw=0.8)
    ax.set_xlabel("ceiling$^2$"); ax.set_ylabel("stored c_full")
    ax.set_title(f"c_full $\\equiv$ ceiling$^2$\nmax |residual| "
                 f"{np.abs(cf - ce**2).max():.1e}", fontsize=6.4)
    lab(ax, "a")

    ax = axes[1]
    pc = np.array([d["pcc"] for d in X]); sk = np.array([d["skill"] for d in X])
    ax.scatter(pc / ce, sk, s=16, color=P["teal"], zorder=3)
    lo, hi = min(sk.min(), (pc / ce).min()) * 0.98, max(sk.max(), (pc / ce).max()) * 1.02
    ax.plot([lo, hi], [lo, hi], "--", color=P["grey_m"], lw=0.8)
    ax.set_xlabel("pcc / ceiling"); ax.set_ylabel("stored skill")
    ax.set_title(f"skill $\\equiv$ pcc / ceiling\nmax |residual| "
                 f"{np.abs(sk - pc/ce).max():.1e}", fontsize=6.4)
    lab(ax, "b")

    ax = axes[2]
    ax.axis("off")
    ax.text(0, 1, "Two σ estimators — never compare their spans\n\n"
            "A  ladder matching\n"
            "     blur the measured truth to level t, then find the σ at which\n"
            "     truth-vs-blurred-truth correlation equals the method's PCC.\n"
            "     Used by: Fig 1e, Fig 2b, Fig 3c/f/g, Fig 5c.\n\n"
            "B  band-pass + τ\n"
            "     band B_i = S_(t-1) − S_t; ER(τ) = smallest σ whose band-wise\n"
            "     PCC still reaches τ, interpolated in log σ.\n"
            "     Used by: Fig 1f.\n\n"
            "Both are inverses of a per-section ladder, so a value is only\n"
            "comparable to another computed on the SAME ladder and target.",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.4,
            linespacing=1.6, color=P["black"], family="monospace")
    lab(ax, "c", x=-0.02)
    fig.tight_layout(pad=0.7)
    save(fig, "S1_estimator_definitions")


# ───────────────── S2：官方 PCC 复现 + 低分方法不可解析 ─────────────────
def s2():
    L = jl(f"{RES}/hest_ladder.json")
    enc = sorted(os.path.basename(f)[len("hest_reported_pcc_"):-5]
                 for f in glob.glob(f"{RES}/hest_reported_pcc_*.json"))
    enc = [e for e in enc if e]

    def eq_of(s_, p_):
        d = L[s_]; sig = {int(k): v for k, v in d["sigma_um"].items()}
        lad = {int(k): v for k, v in d["ladder"].items()}
        pts = [(sig[c], lad[c]) for c in sorted(sig)]
        if p_ >= pts[0][1]:
            return "left"
        for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
            if v0 >= p_ >= v1:
                return "ok"
        return "right"

    rows = []
    for e in enc:
        d = jl(f"{RES}/hest_reported_pcc_{e}.json") or {}
        per = {k: v["pcc"] for k, v in d.items()
               if isinstance(v, dict) and "pcc" in v}
        ids = [i for i in per if i in L]
        fl = [eq_of(i, per[i]) for i in ids]
        rows.append((e, float(np.mean(list(per.values()))),
                     fl.count("ok"), fl.count("right"), len(ids)))
    rows.sort(key=lambda r: -r[1])

    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 62 * MM),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    y = np.arange(len(rows))[::-1]
    frac = [100 * r[2] / r[4] for r in rows]
    cols = [P["red"] if r[0] == "bleep" else P["blue"] for r in rows]
    ax.barh(y, frac, color=cols, edgecolor=P["grey_d"], lw=0.4, height=0.68)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=5.0)
    ax.set_xlabel("% of 72 samples with a resolvable $\\sigma$")
    ax.axvline(50, color=P["grey_m"], lw=0.8, ls="--")
    ax.set_title("Estimator A cannot place low-scoring methods\non the resolution axis",
                 fontsize=6.4)
    lab(ax, "a", x=-0.30)

    ax = axes[1]
    pcc = np.array([r[1] for r in rows]); res = np.array([100 * r[2] / r[4] for r in rows])
    ax.scatter(pcc, res, s=18, color=[P["red"] if r[0] == "bleep" else P["blue"]
                                      for r in rows], zorder=3)
    b1, b0 = np.polyfit(pcc, res, 1)
    xx = np.linspace(pcc.min(), pcc.max(), 20)
    ax.plot(xx, b0 + b1 * xx, "-", color=P["grey_d"], lw=1.0, zorder=2)
    r_, _ = pearsonr(pcc, res)
    for r in rows:
        if r[0] == "bleep":
            ax.annotate("BLEEP", (r[1], 100 * r[2] / r[4]), textcoords="offset points",
                        xytext=(6, -2), fontsize=5.4, color=P["red"])
    ax.set_xlabel("mean reported PCC"); ax.set_ylabel("% resolvable")
    ax.set_title(f"Censoring tracks the score itself\n$r$ = {r_:.3f}, n = {len(rows)}",
                 fontsize=6.4)
    ax.text(0.5, -0.32, "so a σ computed on the resolvable subset flatters\n"
            "the worse methods; we do not report one",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.2,
            linespacing=1.4, color=P["grey_d"])
    lab(ax, "b", x=-0.22)
    fig.tight_layout(pad=0.8)
    save(fig, "S2_estimator_censoring")


# ───────────────────────── S12：各 n 的推断下限 ─────────────────────────
def s12():
    fig, ax = plt.subplots(figsize=(120 * MM, 62 * MM))
    ns = list(range(2, 17))
    fl = [signp(n, n) for n in ns]
    ax.plot(ns, fl, "o-", color=P["blue"], lw=1.3, ms=4)
    ax.set_yscale("log")
    ax.axhline(0.05, color=P["grey_m"], lw=0.9, ls="--")
    ax.text(16, 0.05, " α = 0.05", va="center", fontsize=5.6, color=P["grey_m"])
    for n_, lb, dx, dy in ((2, "n = 2  Visium HD sweeps", 6, -14),
                           (7, "n = 7  Xenium specimens", 6, 12),
                           (8, "n = 8  Xenium (all)", 24, -2),
                           (10, "n = 10  HEST cohorts", 6, -16)):
        ax.scatter([n_], [signp(n_, n_)], s=34, facecolor="none",
                   edgecolor=P["red"], lw=1.1, zorder=5)
        ax.annotate(f"{lb}\nP $\\geq$ {signp(n_, n_):.4f}", (n_, signp(n_, n_)),
                    textcoords="offset points", xytext=(dx, dy), fontsize=5.0,
                    color=P["red"], linespacing=1.35, ha="left")
    ax.set_xlabel("number of independent units")
    ax.set_ylabel("smallest attainable two-sided P")
    ax.set_title("What each design can possibly show\n"
                 "(exact sign test, all units agreeing)", fontsize=6.6)
    ax.text(0.98, 0.95, "Holm over 120 pairs needs P $\\leq$ 4.2×10$^{-4}$;\n"
            "at n = 10 the floor is 2.0×10$^{-3}$ — no pair can pass.",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.2,
            linespacing=1.45, color=P["grey_d"])
    fig.tight_layout(pad=0.7)
    save(fig, "S12_inference_floors")


# ───────────────────────── S13：数据集全图 ─────────────────────────
def s13():
    L = jl(f"{RES}/hest_ladder.json") or {}
    X = [jl(f) for f in sorted(glob.glob(f"{RES}/xenium/*.json"))]
    X = [d for d in X if d]
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 64 * MM),
                             gridspec_kw={"width_ratios": [1, 1.1]})

    ax = axes[0]
    coh = {}
    for v in L.values():
        coh.setdefault(v["cohort"], {"Visium": 0, "Xenium": 0})[v["platform"]] += 1
    order = sorted(coh, key=lambda c: -sum(coh[c].values()))
    y = np.arange(len(order))[::-1]
    vv = np.array([coh[c]["Visium"] for c in order], float)
    xx = np.array([coh[c]["Xenium"] for c in order], float)
    ax.barh(y, vv, color=P["blue"], edgecolor=P["grey_d"], lw=0.4, height=0.66,
            label="Visium")
    ax.barh(y, xx, left=vv, color=P["teal"], edgecolor=P["grey_d"], lw=0.4,
            height=0.66, hatch="///", label="Xenium")
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=5.4)
    ax.set_xlabel("samples")
    ax.set_title(f"HEST breadth line\n{len(L)} samples, {len(order)} cohorts",
                 fontsize=6.4)
    ax.legend(fontsize=5.2, loc="lower right", handlelength=1.1)
    lab(ax, "a", x=-0.30)

    ax = axes[1]
    sp = {}
    for d in X:
        sp.setdefault(specimen(d["name"]), []).append(d)
    ks = sorted(sp, key=lambda k: -len(sp[k]))
    y = np.arange(len(ks))[::-1]
    ax.barh(y, [len(sp[k]) for k in ks], color=P["violet"], edgecolor=P["grey_d"],
            lw=0.4, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([k.replace("Human_Breast_Biomarkers_", "Breast ")
                        .replace("Xenium_", "").replace("_", " ")[:30] for k in ks],
                       fontsize=5.0)
    ax.set_xlabel("section regions contributed")
    ax.set_xticks(range(0, 4))
    ax.set_title(f"Xenium depth line\n{len(X)} regions from {len(ks)} independent "
                 f"specimens", fontsize=6.4)
    ax.text(0.97, 0.05, "every region-level count in the paper\n"
            "is pseudo-replicated at this ratio",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.0,
            linespacing=1.4, color=P["grey_d"])
    lab(ax, "b", x=-0.52)
    fig.tight_layout(pad=0.8)
    save(fig, "S13_cohort_map")


# ───────────────────────── S3：噪声天花板全貌 ─────────────────────────
def s3():
    X = [d for d in (jl(f) for f in sorted(glob.glob(f"{RES}/xenium/*.json"))) if d]
    nm = [d["name"] for d in X]
    pc = np.array([d["pcc"] for d in X]); ce = np.array([d["ceiling"] for d in X])
    cm = np.array([d.get("counts_median", np.nan) for d in X])
    NSP = len({specimen(n) for n in nm})
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 56 * MM))

    ax = axes[0]
    o = np.argsort(pc)
    irr = (1 - ce)[o]; mod = (ce - pc)[o]
    yy = np.arange(len(o))
    ax.barh(yy, mod, color=P["blue"], edgecolor=P["grey_d"], lw=0.35, height=0.72,
            label="model error")
    ax.barh(yy, irr, left=mod, color=P["grey_m"], edgecolor=P["grey_d"], lw=0.35,
            height=0.72, label="irreducible noise")
    ax.set_yticks([]); ax.set_ylabel(f"{len(X)} regions", fontsize=6)
    ax.set_xlabel("1 − PCC, decomposed", fontsize=6)
    ax.set_title("Most of the gap is model error,\nbut not all of it", fontsize=6.4)
    ax.legend(fontsize=5.0, loc="lower right", handlelength=1.1)
    lab(ax, "a", x=-0.14)

    ax = axes[1]
    fr = 100 * (1 - ce) / (1 - pc)
    ax.hist(fr, bins=8, color=P["grey_l"], edgecolor=P["grey_d"], lw=0.5)
    ax.axvline(np.median(fr), color=P["red"], lw=1.2)
    ax.set_xlabel("irreducible share of the gap (%)", fontsize=6)
    ax.set_ylabel("regions", fontsize=6)
    ax.set_title(f"median {np.median(fr):.0f}%  (range {fr.min():.0f}–{fr.max():.0f}%)",
                 fontsize=6.4)
    lab(ax, "b", x=-0.22)

    ax = axes[2]
    m = np.isfinite(cm)
    ax.scatter(cm[m], ce[m], s=16, color=P["blue"], zorder=3)
    ax.set_xscale("log")
    r_, _ = pearsonr(np.log10(cm[m]), ce[m])
    ax.set_xlabel("median transcripts per bin", fontsize=6)
    ax.set_ylabel("split-half ceiling", fontsize=6)
    ax.set_title(f"Ceiling tracks counts\n$r$ = {r_:+.3f} (n = {int(m.sum())} regions)",
                 fontsize=6.4)
    lab(ax, "c", x=-0.24)
    fig.tight_layout(pad=0.8)
    save(fig, "S3_noise_ceiling")


# ───────────────────────── S6：缓冲区与泄漏全貌 ─────────────────────────
def s6():
    B = jl(f"{RES}/buffer_cv.json")
    if not B:
        print("  S6 跳过：缺 buffer_cv"); return
    DS = [0, 25, 50, 100, 200, 400, 800]
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 56 * MM))

    ax = axes[0]
    for g, col, mk in ((4, P["blue"], "o"), (8, P["blue2"], "s"), (16, P["teal"], "^")):
        for s_, ls in zip(sorted(B), ("-", "--")):
            c = np.array([B[s_][f"g{g}_d{d_}"]["pcc"] for d_ in DS])
            ax.plot(DS, 100 * (c / c[0] - 1), mk + ls, color=col, lw=1.0, ms=2.6,
                    label=f"{g}×{g} {s_[-2:]}")
    ax.axhline(0, color=P["grey_m"], lw=0.8)
    ax.set_xlabel("buffer (µm)", fontsize=6); ax.set_ylabel("PCC change (%)", fontsize=6)
    ax.set_title("Per section and grid", fontsize=6.4)
    ax.legend(fontsize=4.4, ncol=2, loc="lower left", handlelength=1.1)
    lab(ax, "a", x=-0.20)

    ax = axes[1]
    rs = np.mean([B[s_]["random_spot"] for s_ in B])
    bars, labs_ = [], []
    for g in (4, 8, 16):
        b0 = np.mean([B[s_][f"g{g}_d0"]["pcc"] for s_ in B])
        b8 = np.mean([B[s_][f"g{g}_d800"]["pcc"] for s_ in B])
        bars.append((rs - b0, b0 - b8)); labs_.append(f"{g}×{g}")
    x = np.arange(3)
    ax.bar(x, [b[0] for b in bars], color=P["red"], edgecolor=P["grey_d"], lw=0.4,
           width=0.55, label="random → block")
    ax.bar(x, [b[1] for b in bars], bottom=[b[0] for b in bars], color=P["blue"],
           edgecolor=P["grey_d"], lw=0.4, width=0.55, label="block → +800 µm buffer")
    ax.set_xticks(x); ax.set_xticklabels(labs_, fontsize=6)
    ax.set_ylabel("ΔPCC", fontsize=6)
    ax.set_title("Leakage budget: the buffer adds\nlittle on top of blocking",
                 fontsize=6.4)
    ax.legend(fontsize=4.8, loc="upper left", handlelength=1.1)
    for i, b in enumerate(bars):
        ax.text(i, b[0] + b[1] + 0.004, f"{100*b[1]/(b[0]+b[1]):.0f}%", ha="center",
                fontsize=5.2, color=P["blue"])
    lab(ax, "b", x=-0.22)

    ax = axes[2]
    for g, col, mk in ((4, P["blue"], "o"), (8, P["blue2"], "s"), (16, P["teal"], "^")):
        tl, dp = [], []
        for d_ in DS[1:]:
            tf = np.mean([B[s_][f"g{g}_d{d_}"]["train_frac"] for s_ in B])
            c0 = np.mean([B[s_][f"g{g}_d0"]["pcc"] for s_ in B])
            cd = np.mean([B[s_][f"g{g}_d{d_}"]["pcc"] for s_ in B])
            tl.append(100 * (1 - tf)); dp.append(c0 - cd)
        ax.plot(tl, dp, mk + "-", color=col, lw=1.0, ms=3, label=f"{g}×{g}")
    ax.set_xlabel("training spots removed (%)", fontsize=6)
    ax.set_ylabel("ΔPCC vs no buffer", fontsize=6)
    ax.set_title("Not simply training-set shrinkage", fontsize=6.4)
    ax.legend(fontsize=5.0, loc="upper left", handlelength=1.1)
    lab(ax, "c", x=-0.24)
    fig.tight_layout(pad=0.8)
    save(fig, "S6_buffer_leakage")


# ───────────────────── S11：排名一致性的留一法内部 ─────────────────────
def s11():
    fig, axes = plt.subplots(1, 2, figsize=(150 * MM, 58 * MM))
    for ax, ff, tag, letter in ((axes[0], "method_rank_cross", "cross-section LOO", "a"),
                                (axes[1], "method_rank_within", "within-section block CV", "b")):
        d = jl(f"{RES}/{ff}.json")
        if not d:
            continue
        rr = d["rows"]
        pcc = np.array([r[1] for r in rr]); sig = np.array([r[2] for r in rr])
        base = spearmanr(pcc, -sig).statistic
        jk = []
        for i in range(len(rr)):
            k = [j for j in range(len(rr)) if j != i]
            jk.append(spearmanr(pcc[k], -sig[k]).statistic)
        ax.axhline(base, color=P["red"], lw=1.2, label=f"all {len(rr)}: $\\rho$ = {base:.3f}")
        ax.scatter(range(len(jk)), jk, s=20, color=P["blue"], zorder=3)
        ax.set_xticks(range(len(rr)))
        ax.set_xticklabels([f"−{r[0][:8]}" for r in rr], rotation=45, ha="right",
                           fontsize=4.8)
        ax.set_ylabel("Spearman $\\rho$ (PCC vs $\\sigma$ rank)", fontsize=6)
        ax.set_ylim(min(jk + [base]) - 0.12, 1.06)
        ax.set_title(f"{tag}\nleave-one-method-out: {min(jk):.3f}–{max(jk):.3f}",
                     fontsize=6.4)
        ax.legend(fontsize=5.0, loc="lower right", handlelength=1.2)
        lab(ax, letter, x=-0.20)
    fig.tight_layout(pad=0.8)
    save(fig, "S11_rank_concordance_internals")


# ───────────── S4：Visium HD 的基因层面混杂（n=2 切片，描述性）─────────────
def s4():
    G = jl(f"{RES}/per_gene_hibou_l.json")
    if not G:
        print("  S4 跳过"); return
    gn = sorted(G)
    pcc = np.array([G[g]["pcc"] for g in gn]); mor = np.array([G[g]["moran"] for g in gn])
    eq = np.array([G[g]["eq"] for g in gn]); ce = np.array([G[g]["ceil"] for g in gn])
    mp = np.array([G[g]["moran_pred"] for g in gn])
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 58 * MM))

    ax = axes[0]
    q = np.nanpercentile(mor, [25, 50, 75])
    sel = [mor <= q[0], (mor > q[0]) & (mor <= q[1]),
           (mor > q[1]) & (mor <= q[2]), mor > q[2]]
    x = np.arange(4); w = 0.26
    for i, (arr, c, l_, ht) in enumerate((
            ([np.nanmedian(pcc[m]) for m in sel], P["blue"], "per-gene PCC", ""),
            ([np.nanmedian(ce[m]) for m in sel], P["grey_m"], "noise ceiling", "///"),
            ([np.nanmedian(pcc[m] / np.where(ce[m] > .05, ce[m], np.nan))
              for m in sel], P["teal"], "PCC / ceiling", "..."))):
        ax.bar(x + (i - 1) * w, arr, w, color=c, edgecolor=P["grey_d"], lw=0.35,
               label=l_, hatch=ht)
    ax.set_xticks(x)
    ax.set_xticklabels(["Q1\nleast", "Q2", "Q3", "Q4\nmost"], fontsize=5.6)
    ax.set_xlabel("Moran's $I$ quartile", fontsize=6)
    ax.set_ylabel("median value", fontsize=6)
    r1 = np.nanmedian(pcc[sel[3]]) / np.nanmedian(pcc[sel[0]])
    r3 = (np.nanmedian(pcc[sel[3]] / ce[sel[3]]) /
          np.nanmedian(pcc[sel[0]] / ce[sel[0]]))
    ax.set_title(f"PCC spreads {r1:.2f}× across quartiles;\n"
                 f"only {r3:.2f}× after dividing by the ceiling", fontsize=6.4)
    ax.legend(fontsize=5.0, loc="upper left", handlelength=1.1)
    lab(ax, "a", x=-0.20)

    ax = axes[1]
    ax.scatter(mor, mp, s=7, color=P["red"], alpha=0.6, zorder=3)
    lo, hi = 0, max(mp.max(), mor.max()) * 1.03
    ax.plot([lo, hi], [lo, hi], "--", color=P["grey_m"], lw=0.8)
    ax.set_xlabel("measured Moran's $I$", fontsize=6)
    ax.set_ylabel("predicted Moran's $I$", fontsize=6)
    ax.set_title(f"{int((mp>mor).sum())}/{len(gn)} genes above identity;\n"
                 f"SD collapses {mor.std()/mp.std():.1f}×", fontsize=6.4)
    lab(ax, "b", x=-0.22)

    ax = axes[2]
    ks = [10, 20, 30, 50, 75, 100, 150]
    ot = np.argsort(-mor); op = np.argsort(-mp)
    rec = [len(set(ot[:k]) & set(op[:k])) / k for k in ks]
    ax.plot(ks, [100 * r for r in rec], "o-", color=P["blue"], lw=1.3, ms=3.5,
            label="recovered")
    ax.plot(ks, [100 * k / len(gn) for k in ks], "--", color=P["grey_m"], lw=1.0,
            label="chance")
    ax.set_xlabel("list size $k$", fontsize=6)
    ax.set_ylabel("% of true top-$k$ recovered", fontsize=6)
    ax.set_title(f"Spatially-variable-gene lists do not\nsurvive: {100*rec[1]:.0f}% at "
                 f"$k$ = 20", fontsize=6.4)
    ax.legend(fontsize=5.2, loc="upper left", handlelength=1.3)
    lab(ax, "c", x=-0.22)
    fig.suptitle("Gene-level confound on 2 Visium HD sections — descriptive; "
                 "the 200 genes are not independent units", fontsize=6.2, y=1.06)
    fig.tight_layout(pad=0.8)
    save(fig, "S4_gene_level_confound")


# ───────────── S5：Moran 结论的范围敏感性 ─────────────
def s5():
    X = [d for d in (jl(f) for f in sorted(glob.glob(f"{RES}/xenium/*.json"))) if d]
    nm = [d["name"] for d in X]
    mt = np.array([d["moran_true"] for d in X]); pc = np.array([d["pcc"] for d in X])
    sp = {}
    for n, a_, b_ in zip(nm, mt, pc):
        sp.setdefault(specimen(n), []).append((a_, b_))
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 56 * MM))

    ax = axes[0]
    # 同时给区域层级与样本层级；区域层级是伪重复，只作参考。
    subs = [("all", np.arange(len(X))),
            ("breast", np.array([i for i, n in enumerate(nm)
                                 if n.startswith("Human_Breast")])),
            ("non-breast", np.array([i for i, n in enumerate(nm)
                                     if not n.startswith("Human_Breast")]))]
    rr_, rs_, ls = [], [], []
    for l_, idx in subs:
        g = {}
        for i in idx:
            g.setdefault(specimen(nm[i]), []).append((mt[i], pc[i]))
        v = np.array([[np.median([x[0] for x in vv]), np.median([x[1] for x in vv])]
                      for vv in g.values()])
        rr_.append(pearsonr(mt[idx], pc[idx])[0])
        rs_.append(pearsonr(v[:, 0], v[:, 1])[0] if len(v) > 2 else np.nan)
        ls.append(f"{l_}\n{len(idx)} reg / {len(v)} spec")
    x = np.arange(3); w = 0.35
    ax.bar(x - w/2, rr_, w, color=P["grey_l"], edgecolor=P["grey_d"], lw=0.4,
           label="region level (pseudo-replicated)")
    ax.bar(x + w/2, rs_, w, color=P["blue"], edgecolor=P["grey_d"], lw=0.4,
           label="specimen level")
    ax.axhline(0, color=P["grey_m"], lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(ls, fontsize=5.0)
    ax.set_ylabel("$r$(PCC, measured Moran's $I$)", fontsize=6)
    for xi, a_, b_ in zip(x, rr_, rs_):
        ax.text(xi - w/2, a_ + 0.03, f"{a_:.2f}", ha="center", fontsize=5.0,
                color=P["grey_d"])
        ax.text(xi + w/2, b_ + 0.03, f"{b_:.2f}", ha="center", fontsize=5.0,
                color=P["blue"])
    ax.set_ylim(0, 1.18)
    ax.set_title("The relation holds in every subset\nexamined", fontsize=6.4)
    ax.legend(fontsize=4.6, loc="lower left", handlelength=1.1)
    ax.text(0.5, -0.40, "but each within-tissue subset has only 4 specimens;\n"
            "the exact permutation floor there is P = 0.083",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.0,
            linespacing=1.4, color=P["grey_d"])
    lab(ax, "a", x=-0.24)

    ax = axes[1]
    ks = sorted(sp)
    jk = []
    for k_ in ks:
        idx = np.array([i for i, n in enumerate(nm) if specimen(n) != k_])
        jk.append(pearsonr(mt[idx], pc[idx])[0])
    base = pearsonr(mt, pc)[0]
    ax.axhline(base, color=P["red"], lw=1.2, label=f"all: $r$ = {base:.3f}")
    ax.scatter(range(len(jk)), jk, s=20, color=P["blue"], zorder=3)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([k.replace("Human_Breast_Biomarkers_", "Br ")
                        .replace("Xenium_", "")[:12] for k in ks],
                       rotation=45, ha="right", fontsize=4.6)
    ax.set_ylabel("$r$ without that specimen", fontsize=6)
    ax.set_title(f"Leave-one-specimen-out:\n{min(jk):.3f}–{max(jk):.3f}", fontsize=6.4)
    ax.legend(fontsize=5.0, loc="lower right", handlelength=1.2)
    lab(ax, "b", x=-0.22)

    ax = axes[2]
    for tag, idx, col, mk in (("16 regions / 8 specimens", np.arange(len(X)), P["blue"], "o"),
                              ("15 / 7 (Fig 3 cohort)",
                               np.array([i for i, n in enumerate(nm)
                                         if "Cervical" not in n]), P["teal"], "s")):
        smt = {}
        for i in idx:
            smt.setdefault(specimen(nm[i]), []).append((mt[i], pc[i]))
        v = np.array([[np.median([x[0] for x in vv]), np.median([x[1] for x in vv])]
                      for vv in smt.values()])
        ax.scatter(v[:, 0], v[:, 1], s=20, color=col, marker=mk, zorder=3,
                   label=f"{tag}: $r$ = {pearsonr(v[:,0], v[:,1])[0]:.3f}")
    ax.set_xlabel("measured Moran's $I$ (specimen median)", fontsize=6)
    ax.set_ylabel("PCC (specimen median)", fontsize=6)
    ax.set_title("Cohort choice barely moves it", fontsize=6.4)
    ax.legend(fontsize=5.0, loc="upper left", handlelength=1.1)
    lab(ax, "c", x=-0.24)
    fig.tight_layout(pad=0.8)
    save(fig, "S5_moran_scope_sensitivity")


# ───────────── S7：iStar 的骨干审计 ─────────────
def s7():
    R = jl(f"{RES}/ruler_fold.json")
    M = {os.path.basename(f)[len("istar_matched_"):-5]: jl(f)
         for f in sorted(glob.glob(f"{RES}/istar_matched_*.json"))}
    if not R or not M:
        print("  S7 跳过"); return
    folds = sorted(M)
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 58 * MM))

    ax = axes[0]
    # 两个比较各自在**自己文件内**做差 —— 两个文件的 iStar 数字不同
    # （0.6404 vs 0.5026，基因集/归一化不同），跨文件相减无意义。
    un = [R[f]["istar_pcc"] - R[f]["scores"]["R_ridgeHEST"] for f in folds]
    ma = [M[f]["istar_official"] - max(M[f]["Ridge_HIPT"], M[f]["kNN_HIPT"])
          for f in folds]
    x = np.arange(len(folds)); w = 0.36
    ax.bar(x - w/2, un, w, color=P["grey_m"], edgecolor=P["grey_d"], lw=0.4,
           label="iStar − Ridge (unmatched backbone)")
    ax.bar(x + w/2, ma, w, color=P["blue"], edgecolor=P["grey_d"], lw=0.4,
           label="iStar − best baseline (matched HIPT)")
    ax.axhline(0, color=P["grey_m"], lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(folds, rotation=30, ha="right", fontsize=5.2)
    ax.set_ylabel("ΔPCC", fontsize=6)
    nflip = sum(1 for a_, b_ in zip(un, ma) if (a_ > 0) != (b_ > 0))
    ax.set_title(f"Matching the encoder flips the verdict\nin {nflip}/{len(folds)} folds",
                 fontsize=6.4)
    ax.legend(fontsize=4.6, loc="lower center", handlelength=1.1,
              bbox_to_anchor=(0.5, -0.62), ncol=1)
    lab(ax, "a", x=-0.22)

    ax = axes[1]
    x = np.arange(len(folds)); w = 0.26
    for i, (k, c, ht) in enumerate((("istar_official", P["red"], ""),
                                    ("Ridge_HIPT", P["blue"], "///"),
                                    ("kNN_HIPT", P["teal"], "..."))):
        ax.bar(x + (i - 1) * w, [M[f][k] for f in folds], w, color=c,
               edgecolor=P["grey_d"], lw=0.35, label=k.replace("_", " "), hatch=ht)
    ax.set_xticks(x); ax.set_xticklabels(folds, rotation=30, ha="right", fontsize=5.2)
    ax.set_ylabel("per-gene PCC", fontsize=6)
    nw = sum(1 for f in folds
             if M[f]["istar_official"] > max(M[f]["Ridge_HIPT"], M[f]["kNN_HIPT"]))
    ax.set_title(f"On shared HIPT features iStar wins\n{nw}/{len(folds)} folds",
                 fontsize=6.4)
    ax.legend(fontsize=4.8, loc="upper right", handlelength=1.1)
    ax.text(0.02, 0.02, "stored `win` field says 4/4 and is wrong;\nrecomputed from the PCCs",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=4.6,
            color=P["grey_m"], linespacing=1.3)
    lab(ax, "b", x=-0.20)

    ax = axes[2]
    ck = [f for f in folds if "checker" in f]; hf = [f for f in folds if "half" in f]
    for tag, col, mk in (("iStar", P["red"], "o"), ("Ridge_HIPT", P["blue"], "s")):
        key = "istar_official" if tag == "iStar" else "Ridge_HIPT"
        drop = [M[c][key] - M[h][key] for c, h in zip(sorted(ck), sorted(hf))]
        ax.plot(range(len(drop)), drop, mk + "-", color=col, lw=1.2, ms=4, label=tag)
    ax.set_xticks(range(len(ck)))
    ax.set_xticklabels([c.split("_")[0] for c in sorted(ck)], fontsize=5.6)
    ax.set_ylabel("PCC drop, checker → half", fontsize=6)
    ax.set_title("Fragility to fold geometry is\nshared, not iStar-specific", fontsize=6.4)
    ax.legend(fontsize=5.0, loc="upper left", handlelength=1.2)
    lab(ax, "c", x=-0.24)
    fig.tight_layout(pad=0.8)
    save(fig, "S7_istar_backbone_audit")


# ───────────── S8：排行榜稳健性 ─────────────
def s8():
    enc = sorted(os.path.basename(f)[len("hest_reported_pcc_"):-5]
                 for f in glob.glob(f"{RES}/hest_reported_pcc_*.json"))
    enc = [e for e in enc if e]
    D = {}
    for e in enc:
        d = jl(f"{RES}/hest_reported_pcc_{e}.json") or {}
        D[e] = {k: v["pcc"] for k, v in d.items()
                if isinstance(v, dict) and "pcc" in v}
    ref = jl(f"{RES}/hest_reported_pcc_phikon_v2.json") or {}
    coh = {k: v["cohort"] for k, v in ref.items()
           if isinstance(v, dict) and "cohort" in v}
    ids = sorted(set.intersection(*[set(D[e]) for e in enc]))
    cohs = sorted(set(coh.values()))
    raw = np.array([np.mean([D[e][i] for i in ids]) for e in enc])
    bal = np.array([np.mean([np.mean([D[e][i] for i in ids if coh[i] == c])
                             for c in cohs]) for e in enc])
    big = [c for c in cohs if sum(1 for i in ids if coh[i] == c) >= 20]
    drop = np.array([np.mean([D[e][i] for i in ids if coh[i] not in big])
                     for e in enc])
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 58 * MM))

    for ax, alt, tag, letter in ((axes[0], bal, "cohort-balanced", "a"),
                                 (axes[1], drop, f"drop {'/'.join(big)}", "b")):
        r1 = np.argsort(np.argsort(-raw)); r2 = np.argsort(np.argsort(-alt))
        for i in range(len(enc)):
            ax.plot([0, 1], [r1[i], r2[i]], "-", color=P["grey_l"], lw=0.7, zorder=1)
        ax.scatter([0] * len(enc), r1, s=12, color=P["blue"], zorder=3)
        ax.scatter([1] * len(enc), r2, s=12, facecolor="none", edgecolor=P["grey_d"],
                   lw=0.8, zorder=3)
        ax.set_xlim(-0.3, 1.3); ax.set_xticks([0, 1])
        ax.set_xticklabels(["raw mean", tag], fontsize=5.6)
        ax.set_ylabel("rank (0 = best)", fontsize=6); ax.invert_yaxis()
        ax.set_title(f"$\\rho$ = {spearmanr(raw, alt).statistic:.3f};  "
                     f"max shift {int(np.abs(r1-r2).max())}", fontsize=6.4)
        lab(ax, letter, x=-0.22)

    ax = axes[2]
    per = np.array([[np.argsort(np.argsort(-np.array([D[e][i] for e in enc])))[j]
                     for i in ids] for j in range(len(enc))])
    o = np.argsort(-raw)
    ax.boxplot([per[j] for j in o], vert=True, widths=0.6, showfliers=False,
               medianprops=dict(color=P["red"], lw=1.0),
               boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    ax.set_xticks(range(1, len(enc) + 1))
    ax.set_xticklabels([enc[j] for j in o], rotation=90, fontsize=4.4)
    ax.set_ylabel("per-sample rank", fontsize=6)
    ax.set_title("Every pipeline spans nearly the whole\nrank range sample to sample",
                 fontsize=6.4)
    lab(ax, "c", x=-0.16)
    fig.tight_layout(pad=0.8)
    save(fig, "S8_leaderboard_robustness")


# ───────────── S9：归一化的机制 ─────────────
def s9():
    enc = sorted(os.path.basename(f)[len("hest_cp10k_"):-5]
                 for f in glob.glob(f"{RES}/hest_cp10k_*.json"))
    A = {e: jl(f"{RES}/hest_reported_pcc_{e}.json") for e in enc}
    B = {e: jl(f"{RES}/hest_cp10k_{e}.json")["per_sample_pcc"] for e in enc}
    ids = sorted(set(k for k, v in A[enc[0]].items()
                     if isinstance(v, dict) and "pcc" in v) & set(B[enc[0]]))
    coh = {i: A[enc[0]][i]["cohort"] for i in ids}
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 56 * MM))

    ax = axes[0]
    xa = np.concatenate([[A[e][i]["pcc"] for i in ids] for e in enc])
    ya = np.concatenate([[B[e][i] for i in ids] for e in enc])
    ax.scatter(xa, ya, s=3, color=P["grey_l"], alpha=0.5, zorder=1)
    k = float((xa * ya).sum() / (xa ** 2).sum())
    xx = np.linspace(0, xa.max(), 20)
    ax.plot(xx, k * xx, "-", color=P["red"], lw=1.2, zorder=3)
    ax.plot(xx, xx, "--", color=P["grey_m"], lw=0.8, zorder=2)
    ax.set_xlabel("log1p PCC", fontsize=6); ax.set_ylabel("CP10K PCC", fontsize=6)
    ax.set_title(f"A proportional shrink:\nCP10K = {k:.3f} × log1p", fontsize=6.4)
    lab(ax, "a", x=-0.22)

    ax = axes[1]
    cohs = sorted(set(coh.values()))
    dc = [np.mean([np.mean([B[e][i] - A[e][i]["pcc"] for i in ids if coh[i] == c])
                   for e in enc]) for c in cohs]
    o = np.argsort(dc)
    ax.barh(range(len(cohs)), [dc[i] for i in o], color=P["red"],
            edgecolor=P["grey_d"], lw=0.4, height=0.66)
    ax.set_yticks(range(len(cohs)))
    ax.set_yticklabels([cohs[i] for i in o], fontsize=5.2)
    ax.axvline(0, color=P["grey_m"], lw=0.8)
    ax.set_xlabel("mean ΔPCC (CP10K − log1p)", fontsize=6)
    ax.set_title(f"All {sum(1 for v in dc if v < 0)}/{len(cohs)} cohorts fall",
                 fontsize=6.4)
    lab(ax, "b", x=-0.34)

    ax = axes[2]
    rat = [np.mean([B[e][i] for i in ids]) / np.mean([A[e][i]["pcc"] for i in ids])
           for e in enc]
    ax.hist(rat, bins=8, color=P["grey_l"], edgecolor=P["grey_d"], lw=0.5)
    ax.axvline(np.mean(rat), color=P["red"], lw=1.2)
    ax.set_xlabel("CP10K / log1p (per encoder)", fontsize=6)
    ax.set_ylabel("encoders", fontsize=6)
    ax.set_title(f"Shrink ratio is near-constant\n{np.mean(rat):.3f} ± "
                 f"{np.std(rat):.3f}", fontsize=6.4)
    ax.text(0.5, -0.34, "σ cost is NOT identifiable: the two targets have\n"
            "different ladders and asymmetric censoring (§37)",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.0,
            linespacing=1.4, color=P["grey_d"])
    lab(ax, "c", x=-0.24)
    fig.tight_layout(pad=0.8)
    save(fig, "S10_placeholder" if False else "S9_normalisation_mechanism")


# ───────────── S10：两个扫描的内部 ─────────────
def s10():
    def sweep(pat, key, extra=None):
        out = []
        for f in sorted(glob.glob(pat)) + ([extra] if extra else []):
            n = os.path.basename(f)[:-5]
            if n.endswith("g3"):
                continue
            m_ = re.search(key + r"(\d+)", f if key == "hvg" else n)
            v = int(m_.group(1)) if m_ else 224
            d = jl(f)
            if not d:
                continue
            sm = d["_summary"]; sl = [k for k in d if not k.startswith("_")]
            out.append(dict(x=v, sm=sm, sl={s_: d[s_] for s_ in sl}))
        return sorted(out, key=lambda r: r["x"])
    CTX = sweep(f"{RES}/ctx_sweep/hibou_l_ctx*.json", "ctx",
                f"{RES}/tower_sweep/hibou_l.json")
    PAN = sweep(f"{RES}/panel_sweep_hvg*/hibou_l.json", "hvg")
    if not CTX or not PAN:
        print("  S10 跳过"); return
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 56 * MM))

    ax = axes[0]
    for v, col, mk, tag in ((CTX, P["violet"], "o", "crop"),
                            (PAN, P["teal"], "s", "panel")):
        cen = []
        for r in v:
            n_c = 0
            for s_, dd in r["sl"].items():
                sig = {int(k): x for k, x in dd["sigma_um"].items()}
                lad = {int(k): dd["ladder"][k]["pcc"] for k in dd["ladder"]}
                cps = sorted(sig)
                pv = dd["methods"]["Ridge_HEST"]["pcc"]
                if pv < lad[cps[-1]]:
                    n_c += 1
            cen.append(100 * n_c / max(len(r["sl"]), 1))
        ax.plot([r["x"] for r in v], cen, mk + "-", color=col, lw=1.2, ms=3.5, label=tag)
    ax.set_xscale("log")
    ax.set_xlabel("knob setting", fontsize=6)
    ax.set_ylabel("% of sections right-censored", fontsize=6)
    ax.set_title("Censoring appears only at the\ncoarse end of the crop sweep",
                 fontsize=6.4)
    ax.legend(fontsize=5.2, loc="upper left", handlelength=1.2)
    lab(ax, "a", x=-0.24)

    ax = axes[1]
    for mk_, c_, lb_ in (("pcc", P["blue"], "PCC"), ("ari", P["teal"], "ARI"),
                         ("dom1", P["violet"], "dom1")):
        v = [np.mean([r["sl"][s_]["methods"]["Ridge_HEST"][mk_] for s_ in r["sl"]])
             for r in PAN]
        v = np.array(v) / v[0]
        ax.plot([r["x"] for r in PAN], v, "o-", color=c_, lw=1.2, ms=3, label=lb_)
    ax.set_xscale("log"); ax.axhline(1, color=P["grey_m"], lw=0.8, ls="--")
    ax.set_xlabel("gene panel size (HVG)", fontsize=6)
    ax.set_ylabel("value relative to HVG 20", fontsize=6)
    ax.set_title("Non-PCC readouts are not monotone\nin panel size", fontsize=6.4)
    ax.legend(fontsize=5.2, loc="lower left", handlelength=1.2)
    lab(ax, "b", x=-0.24)

    ax = axes[2]
    for v, col, mk, tag in ((CTX, P["violet"], "o", "crop 224→1344 px"),
                            (PAN, P["teal"], "s", "panel 20→200 HVG")):
        mo = [np.mean([r["sl"][s_]["moran_true"] for s_ in r["sl"]]) for r in v]
        ax.plot(range(len(v)), np.array(mo) / mo[0], mk + "-", color=col, lw=1.2,
                ms=3.5, label=tag)
    ax.axhline(1, color=P["grey_m"], lw=0.8, ls="--")
    ax.set_xlabel("setting index (coarsest last)", fontsize=6)
    ax.set_ylabel("target Moran's $I$, relative", fontsize=6)
    ax.set_title("Only the panel knob moves the\nprediction target itself", fontsize=6.4)
    ax.legend(fontsize=5.0, loc="lower left", handlelength=1.2)
    lab(ax, "c", x=-0.24)
    fig.tight_layout(pad=0.8)
    save(fig, "S10_sweep_internals")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["1","2","3","4","5","6","7","8","9","10","11","12","13"]
    for k in todo:
        print(f"S{k} …", flush=True)
        globals()[f"s{k}"]()
    print("完成")
