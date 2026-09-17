#!/usr/bin/env python
"""六张投稿级主图，全部从 results/ 下的 JSON 读数，不写死任何数值。

核心结论（每张图只为它服务的那一句）：
    报告分数与物理分辨率之间的换算关系陡峭、随协议而变，并在评测栅格可变时反号——
    排行榜名次因此可被协议选择操纵，而粗输出的下游代价真实存在。

证据链：
    Fig1  σ 可信，且分得开方法；信号功率集中在复现最差的尺度    ← 前提
    Fig2  PCC→σ 换算率陡峭，且因协议而异 3 倍
    Fig3  评测栅格决定「调粗」是加分还是减分，15/15 片反号        ← hero
    Fig4  两个已发表方法在 10/10 癌种不优于冻结特征 + 岭回归
    Fig5  σ 对划分协议的敏感度是 PCC 的 5.5 倍
    Fig6  粗输出的下游代价：六个度量在 16→64 µm 同向劣化

全部文字为英文（投稿要求；且 DejaVu Sans 无 CJK 字形会渲染成空框）。
导出：183 mm 双栏宽，PDF(矢量, fonttype=42) + SVG(可编辑文本) + TIFF(600 dpi)。
"""
import json, glob, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 强制可编辑文本（SVG 保 <text> 节点，PDF 保 TrueType）──
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({
    "font.size": 7, "axes.linewidth": 0.8, "legend.frameon": False,
    "axes.spines.right": False, "axes.spines.top": False,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.titlesize": 7, "axes.titlepad": 9,
})

P = {"blue": "#0F4D92", "blue2": "#3775BA", "red": "#B64342", "teal": "#42949E",
     "violet": "#9A4D8E", "grey_l": "#CFCECE", "grey_m": "#767676", "grey_d": "#4D4D4D",
     "green": "#2E9E44", "black": "#272727"}
MM = 1 / 25.4
RES, OUT = "results", "figures"
BINS = [8, 16, 32, 64]
COH = ["SKCM", "HCC", "LUNG", "PAAD", "COAD", "READ", "IDC", "LYMPH_IDC", "PRAD", "CCRCC"]

# conversion_rate.json 的条目名是中文（内部记录用）；投稿图必须英文。
PROTO_EN = {
    "深度线 · 25 个图像编码器（跨片）": "Deep line\n25 image encoders",
    "方法 · 片内块 CV（旧 200 基因基准）": "Methods\nwithin-section block CV",
    "方法 · 跨片留一（旧 200 基因基准）": "Methods\nleave-one-section-out",
    "多尺度分箱（对照）": "Multi-scale binning\n(control)",
}


def specimen(name):
    """15 个切片区域实为 7 个独立样本：乳腺 S1–S4 各有 Top/Mid/Bot 三个子区域。
    按区域计数的「k/15」是伪重复，图上必须同时给出样本层级的数字。"""
    m = re.match(r"(Human_Breast_Biomarkers_S\d)_(Top|Mid|Bot)$", name)
    return m.group(1) if m else name


def by_specimen(names):
    """name 列表 → {样本: [下标]}"""
    g = {}
    for i, n in enumerate(names):
        g.setdefault(specimen(n), []).append(i)
    return g


def lab(ax, s, x=-0.16, y=1.13):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="left", va="bottom")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUT}/{name}.pdf", flush=True)


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


# ══════════════════════════════ Fig 1 ══════════════════════════════
def fig1():
    """σ 度量的三段论：标定可信 → 分得开方法 → 且量到了要紧的尺度。"""
    d = jload(f"{RES}/effres_hibou_l_hvg50_t2048.json")
    if not d:
        print("  Fig1 跳过：缺 effres_hibou_l"); return
    sig = {int(k): v for k, v in d["sigma_um"].items()}
    cps = sorted(sig)
    sv = np.array([sig[c] for c in cps])
    tp = {int(k): v for k, v in d["truth_power"].items()}
    meths = sorted(d["methods"], key=lambda m: -m["pcc"])
    TAU = "0.2"

    # Archetype: quantitative grid, hero = panel b。三格科学权重不等，
    # 故不做等宽（skill Pattern 15）。
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 57 * MM),
                             gridspec_kw={"width_ratios": [1.0, 1.45, 1.15]})

    # ── a  σ 由 δ 种子扩散的二阶矩实测标定，非解析近似 ──
    ax = axes[0]
    ax.plot(cps, sv, "o-", color=P["blue"], lw=1.2, ms=3)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Diffusion steps $t$"); ax.set_ylabel("Measured $\\sigma$ (µm)")
    sl = np.polyfit(np.log(cps), np.log(sv), 1)[0]
    ax.set_title(f"Calibration by second moment\n($\\sigma \\propto t^{{{sl:.2f}}}$)")
    lab(ax, "a")

    # ── b  带通相关曲线 + τ 交点 = ER；曲线密集是事实，交点把差异说清楚 ──
    ax = axes[1]
    # 高亮按 ER 取两端 —— 与标题里的 span 必须是同两条曲线；
    # 注意 ER 最差的不是 PCC 最差的那个方法，这本身值得在正文点一句。
    _er = {m["method"]: m["eff_res_um"][TAU] for m in meths}
    hi = {min(_er, key=_er.get): P["blue"], "Ridge_HEST": P["teal"],
          max(_er, key=_er.get): P["red"]}
    for m in meths:
        cur = np.array([{int(k): v for k, v in m["curve"].items()}[c] for c in cps])
        c = hi.get(m["method"])
        ax.plot(sv, cur, "-", lw=1.3 if c else 0.7, color=c or P["grey_l"],
                zorder=3 if c else 1, label=m["method"] if c else None)
    er = np.array([m["eff_res_um"][TAU] for m in meths])
    ax.axhline(float(TAU), color=P["grey_m"], lw=0.8, ls="--", zorder=2)
    ax.scatter(er, [float(TAU)] * len(er), s=14, color=P["black"], zorder=5,
               clip_on=False, marker="v")
    ax.set_xscale("log")
    ax.set_xlabel("Band centre $\\sigma$ (µm)"); ax.set_ylabel("Band-wise PCC")
    ax.set_title(f"Equivalent resolution at $\\tau$={TAU}\nspans "
                 f"{er.min():.1f}–{er.max():.1f} µm ({er.max()/er.min():.2f}×, "
                 f"n={len(meths)} methods)")
    ax.legend(fontsize=5.2, loc="lower right", handlelength=1.2)
    # 交点区放大：主图上九条几乎重合，差异只有在 τ 附近才看得见
    ins = ax.inset_axes([0.07, 0.55, 0.42, 0.40])
    for m in meths:
        cur = np.array([{int(k): v for k, v in m["curve"].items()}[c] for c in cps])
        c = hi.get(m["method"])
        ins.plot(sv, cur, "-", lw=1.1 if c else 0.6, color=c or P["grey_l"],
                 zorder=3 if c else 1)
    ins.axhline(float(TAU), color=P["grey_m"], lw=0.7, ls="--")
    ins.scatter(er, [float(TAU)] * len(er), s=8, color=P["black"], zorder=5, marker="v")
    ins.set_xscale("log")
    ins.set_xlim(sv[0] * 0.93, er.max() * 1.5); ins.set_ylim(0.10, 0.32)
    ins.tick_params(labelsize=5.0, length=1.8, pad=1)
    ins.set_xticks([10, 15, 20, 28]); ins.set_xticklabels(["10", "15", "20", "28"])
    for s in ("top", "right"):
        ins.spines[s].set_visible(True)
    for s in ins.spines.values():
        s.set_linewidth(0.5); s.set_color(P["grey_m"])
    lab(ax, "b")

    # ── c  带通功率的尺度分布（带内归一化）vs 该尺度上的复现保真度 ──
    #     truth_power 的分母是未中心化的表达矩阵，含基因均值(DC)，
    #     故原始占比合计仅 ~9.7%；只有带内归一化才是「空间变异的尺度分布」。
    ax = axes[2]
    tot = sum(tp.values())
    sh = np.array([100 * tp[c] / tot for c in cps])
    x = np.arange(len(cps))
    ax.bar(x, sh, color=P["grey_l"], edgecolor=P["grey_d"], lw=0.5, width=0.75, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:.0f}" for s in sv], rotation=45, ha="right", fontsize=5.5)
    ax.set_xlabel("Band centre $\\sigma$ (µm)")
    ax.set_ylabel("Share of band-passed\ntruth power (%)")
    best = np.array([{int(k): v for k, v in meths[0]["curve"].items()}[c] for c in cps])
    ax2 = ax.twinx()
    ax2.plot(x, best, "o-", color=P["blue"], lw=1.3, ms=3, zorder=3)
    ax2.set_ylabel("Band-wise PCC,\nbest method", color=P["blue"])
    ax2.tick_params(axis="y", colors=P["blue"]); ax2.spines["top"].set_visible(False)
    ax2.set_ylim(0, 1)
    # 注释画在 ax2 上：twin 轴层级高于 ax，否则蓝色曲线会盖住文字。
    ax2.annotate(f"{sh[0]:.0f}% of the signal sits here,\n"
                 f"where the best method reaches {best[0]:.2f}",
                 xy=(0.42, sh[0] * 0.97), xycoords=ax.transData,
                 xytext=(1.6, 0.99), textcoords=ax2.transData, fontsize=5.4,
                 color=P["grey_d"], va="top", ha="left", zorder=10,
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.8),
                 arrowprops=dict(arrowstyle="->", lw=0.6, color=P["grey_d"]))
    ax.set_title("Signal concentrates where\nfidelity is lowest")
    lab(ax, "c")

    fig.tight_layout(pad=0.7)
    save(fig, "Fig1_metric_validation")


# ══════════════════════════════ Fig 2 ══════════════════════════════
def fig2():
    rows = []
    for f in sorted(glob.glob(f"{RES}/legacy8k/tower_*.json")):
        n = os.path.basename(f)[len("tower_"):-5]
        if any(x in n for x in ("grid", "ctx")):
            continue
        d = jload(f)
        if not d:
            continue
        sl = [k for k in d if k.startswith("Visium")]
        s = [d[k]["eq_sigma"] for k in sl if d[k].get("flag") == "ok"]
        p = [d[k]["pcc"] for k in sl if d[k].get("flag") == "ok"]
        if s:
            rows.append((n, float(np.mean(p)), float(np.mean(s))))
    conv = jload(f"{RES}/conversion_rate.json") or []
    if not rows:
        print("  Fig2 跳过：缺 25 塔数据"); return

    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 64 * MM),
                             gridspec_kw={"width_ratios": [1.3, 1]})
    # ── a  hero：25 座图像编码器的 PCC–σ 关系 ──
    ax = axes[0]
    pc = np.array([r[1] for r in rows]); sg = np.array([r[2] for r in rows])
    ax.scatter(pc, sg, s=16, color=P["blue"], edgecolor="white", lw=0.4, zorder=3)
    b, a = np.polyfit(pc, np.log(sg), 1)
    xx = np.linspace(pc.min(), pc.max(), 60)
    ax.plot(xx, np.exp(a + b * xx), "-", color=P["red"], lw=1.2, zorder=2)
    r2 = 1 - ((np.log(sg) - (a + b * pc)) ** 2).sum() / \
             ((np.log(sg) - np.log(sg).mean()) ** 2).sum()
    per = (np.exp(0.01 * b) - 1) * 100
    ax.set_yscale("log")
    ax.set_xlabel("Reported PCC"); ax.set_ylabel("Equivalent resolution $\\sigma$ (µm)")
    ax.set_title(f"25 image encoders: +0.01 PCC $\\Rightarrow$ {per:+.1f}% $\\sigma$ "
                 f"($R^2$={r2:.3f}, n={len(rows)})")
    lo, hi = rows[int(np.argmin(pc))], rows[int(np.argmax(pc))]
    ax.annotate(lo[0], (lo[1], lo[2]), textcoords="offset points", xytext=(5, 3),
                fontsize=5.2, color=P["grey_d"], ha="left")
    ax.annotate(hi[0], (hi[1], hi[2]), textcoords="offset points", xytext=(-5, 5),
                fontsize=5.2, color=P["grey_d"], ha="right")
    lab(ax, "a", x=-0.13)

    # ── b  换算率因协议而异 ⇒ 不可跨基准迁移 ──
    ax = axes[1]
    cs = [c for c in conv if isinstance(c.get("pct_per_0.01pcc"), (int, float))]
    if cs:
        cs = sorted(cs, key=lambda c: c["pct_per_0.01pcc"])
        names = [PROTO_EN.get(c["name"], c["name"]) for c in cs]
        vals = np.array([c["pct_per_0.01pcc"] for c in cs])
        ctrl = ["control" in n for n in names]
        col = [P["grey_m"] if k else P["blue"] for k in ctrl]
        y = np.arange(len(cs))[::-1]
        ax.barh(y, vals, color=col, edgecolor=P["grey_d"], lw=0.5, height=0.6)
        ax.axvline(0, color=P["grey_m"], lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=5.3)
        ax.set_xlabel("$\\sigma$ change per +0.01 PCC (%)")
        rng = [v for v, k in zip(vals, ctrl) if not k]
        ax.set_title("Exchange rate is protocol-specific\n"
                     f"({min(rng):.1f}% to {max(rng):.1f}% across benchmarks)")
        ax.set_xlim(vals.min() * 1.32, max(0.6, vals.max() * 1.2))
        for yi, v in zip(y, vals):
            ax.text(v - 0.5, yi, f"{v:+.1f}", va="center", ha="right", fontsize=5.5,
                    color=P["grey_d"])
    lab(ax, "b", x=-0.55)
    fig.tight_layout(pad=0.7)
    save(fig, "Fig2_conversion_rate")


# ══════════════════════════════ Fig 3 (hero) ══════════════════════════════
def fig3():
    pair = []
    for f in sorted(glob.glob(f"{RES}/downstream_*.json")):
        n = os.path.basename(f)[len("downstream_"):-5]
        d = jload(f)
        if not d or not all(str(b) in d.get("scales", {}) for b in BINS):
            continue
        fx = [d["scales"][str(b)]["pcc"] for b in BINS]
        ow = []
        for b in BINS:
            p = f"{RES}/xenium_multi/{n}_bin{b}.json" if b != 16 else f"{RES}/xenium/{n}.json"
            q = jload(p)
            ow.append(q["pcc"] if q else np.nan)
        if np.isfinite(ow).all():
            pair.append((n, np.array(ow), np.array(fx)))
    if not pair:
        print("  Fig3 跳过：缺配对数据"); return
    N = len(pair)
    SPEC = by_specimen([p[0] for p in pair]); NSP = len(SPEC)

    fig = plt.figure(figsize=(183 * MM, 68 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.45)

    # ── a  hero：同一批预测，两种评测栅格，逐片配对 ──
    ax = fig.add_subplot(gs[0])
    for _, ow, fx in pair:
        ax.plot(BINS, ow, "-", color=P["red"], lw=0.6, alpha=0.4)
        ax.plot(BINS, fx, "-", color=P["blue"], lw=0.6, alpha=0.4)
    # marker 冗余编码：两族颜色的灰度亮度为 101 与 66，仅靠色相在黑白打印下偏弱
    ax.plot(BINS, np.median([p[1] for p in pair], 0), "o-", color=P["red"], lw=2, ms=4,
            label="Evaluation grid follows prediction")
    ax.plot(BINS, np.median([p[2] for p in pair], 0), "s-", color=P["blue"], lw=2, ms=4,
            label="Evaluation grid fixed at 16 µm")
    ax.set_xscale("log", base=2); ax.set_xticks(BINS); ax.set_xticklabels(BINS)
    ax.set_xlabel("Prediction bin size (µm)"); ax.set_ylabel("Reported PCC")
    ax.set_title(f"Identical predictions, opposite conclusion\n(n={N} regions from {NSP} specimens)")
    ax.legend(fontsize=5.5, loc="lower left", handlelength=1.4)
    lab(ax, "a", x=-0.13)

    # ── b  8→64 µm 的相对变化，逐片配对 ──
    ax = fig.add_subplot(gs[1])
    so = np.array([100 * (p[1][-1] / p[1][0] - 1) for p in pair])
    sf = np.array([100 * (p[2][-1] / p[2][0] - 1) for p in pair])
    for a_, b_ in zip(so, sf):
        ax.plot([0, 1], [a_, b_], "-", color=P["grey_l"], lw=0.6, zorder=1)
    ax.scatter([0] * N, so, s=12, color=P["red"], marker="o", zorder=3)
    ax.scatter([1] * N, sf, s=12, color=P["blue"], marker="s", zorder=3)
    ax.axhline(0, color=P["grey_m"], lw=0.8, ls="--")
    ax.set_xlim(-0.4, 1.4); ax.set_xticks([0, 1])
    ax.set_xticklabels(["follows", "fixed"], fontsize=6)
    ax.set_xlabel("Evaluation grid")
    ax.set_ylabel("PCC change, 8 → 64 µm (%)")
    rev = int(((so > 0) & (sf < 0)).sum())
    # 区域层级会伪重复（12/15 区域来自 4 个乳腺样本），故同时报样本层级。
    rev_sp = sum(1 for v in SPEC.values()
                 if np.median([so[i] for i in v]) > 0 > np.median([sf[i] for i in v]))
    ax.set_title(f"{rev}/{N} regions reverse sign\n"
                 f"({rev_sp}/{NSP} specimens; median {np.median(so):+.1f}% vs {np.median(sf):+.1f}%)")
    lab(ax, "b", x=-0.42)

    # ── c  固定栅格下 PCC 与 σ 的走向 ──
    ax = fig.add_subplot(gs[2])
    ms = []
    for b in BINS:
        g = glob.glob(f"{RES}/xenium_multi/*_bin{b}.json") if b != 16 \
            else glob.glob(f"{RES}/xenium/*.json")
        v = [q["eq_sigma"] for q in (jload(f) for f in g)
             if q and q.get("eq_flag") == "ok" and np.isfinite(q.get("eq_sigma", np.nan))]
        ms.append(np.median(v) if v else np.nan)
    med_fx = np.median([p[2] for p in pair], 0)
    ax2 = ax.twinx()
    ax.plot(BINS, med_fx, "o-", color=P["blue"], lw=1.4, ms=3.5)
    ax2.plot(BINS, ms, "s--", color=P["grey_d"], lw=1.4, ms=3.5)
    ax.set_xscale("log", base=2); ax.set_xticks(BINS); ax.set_xticklabels(BINS)
    ax.set_xlabel("Prediction bin size (µm)")
    ax.set_ylabel("PCC (fixed grid)", color=P["blue"])
    ax2.set_ylabel("$\\sigma$ (µm)", color=P["grey_d"])
    ax2.spines["top"].set_visible(False)
    ax.tick_params(axis="y", colors=P["blue"]); ax2.tick_params(axis="y", colors=P["grey_d"])
    # 两者的最优点并不相同：PCC 峰在 16 µm，σ 最好的是 8 µm。
    # 也就是说细端 PCC 与 σ 仍然打架（8 µm 分辨率更好但分数更低），
    # 一致的只有 16 µm 之后。标题必须把这两件事分开说，不能笼统写「optimum」。
    opt_p = BINS[int(np.nanargmax(med_fx))]
    opt_s = BINS[int(np.nanargmin(ms))]
    ax.axvline(opt_p, color=P["grey_l"], lw=0.8, zorder=0)
    ax.set_title(f"Fixed grid: PCC peaks at {opt_p} µm,\n"
                 f"$\\sigma$ is best at {opt_s} µm")
    lab(ax, "c", x=-0.42)
    save(fig, "Fig3_evaluation_grid")


# ══════════════════════════════ Fig 4 ══════════════════════════════
def fig4():
    def best(pat):
        """每样本取该方法所有配置里的最好成绩 —— 对已发表方法最有利的读法。"""
        out = {}
        for f in glob.glob(pat):
            d = jload(f) or {}
            for sid, v in d.items():
                if isinstance(v, dict) and isinstance(v.get("pcc"), (int, float)) \
                        and v["pcc"] == v["pcc"]:
                    if sid not in out or v["pcc"] > out[sid][0]:
                        out[sid] = (v["pcc"], v.get("cohort", "?"))
        return out
    M = {"HisToGene": best(f"{RES}/histogene_*.json"),
         "Hist2ST": best(f"{RES}/h2st2_[mp]*.json")}
    rd = jload(f"{RES}/hest_reported_pcc_phikon_v2.json") or {}
    ref = {k: (v["pcc"], v["cohort"]) for k, v in rd.items() if "pcc" in v}
    if not ref or not any(M.values()):
        print("  Fig4 跳过：缺方法数据"); return

    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 64 * MM),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    # ── a  hero：逐癌种成对条形 ──
    ax = axes[0]
    present = [c for c in COH if any(v[1] == c for v in ref.values())]
    x = np.arange(len(present)); w = 0.26
    HATCH = ["", "///", "..."]
    series = [("HisToGene", M["HisToGene"], P["teal"]),
              ("Hist2ST", M["Hist2ST"], P["violet"]),
              ("Ridge + phikon-v2", ref, P["blue"])]
    tab = {}
    for i, (nm, mm, col) in enumerate(series):
        vals = [np.mean([p for p, c in mm.values() if c == ch])
                if any(c == ch for _, c in mm.values()) else np.nan for ch in present]
        tab[nm] = vals
        # hatch：teal 与 violet 的灰度亮度只差 18/255，黑白打印下不可分（skill Pattern 6）
        ax.bar(x + (i - 1) * w, vals, width=w, color=col, edgecolor=P["grey_d"],
               lw=0.4, label=nm, hatch=HATCH[i])
        # 有几个队列该方法的均值≈0 甚至为负，柱子在 0.55 的量程下不可见；
        # 不标数值的话读者无法区分「≈0」与「没跑」。
        for xi, v in zip(x, vals):
            if np.isfinite(v) and abs(v) < 0.02:
                ax.text(xi + (i - 1) * w, 0.012, f"{v:+.3f}", rotation=90,
                        ha="center", va="bottom", fontsize=5.0, color=col)
    nwin = sum(1 for j in range(len(present))
               if max(tab["HisToGene"][j], tab["Hist2ST"][j]) > tab["Ridge + phikon-v2"][j])
    ax.set_xticks(x); ax.set_xticklabels(present, rotation=40, ha="right", fontsize=5.5)
    ax.set_ylabel("Per-gene PCC")
    # SKCM 上 Hist2ST 的队列均值为负（比随机还差）；下界卡 0 会把它静默裁掉。
    _lo = min(np.nanmin(v) for v in tab.values())
    _hi = max(np.nanmax(v) for v in tab.values())
    ax.set_ylim(min(0, _lo * 1.6) - 0.01, _hi * 1.30)
    ax.axhline(0, color=P["grey_m"], lw=0.6, zorder=0)
    ax.set_title("Published methods vs a linear baseline on frozen features\n"
                 f"(baseline higher in {len(present)-nwin}/{len(present)} cohorts)")
    ax.legend(fontsize=5.5, ncol=3, loc="upper center", handlelength=1.2,
              bbox_to_anchor=(0.5, 1.0), columnspacing=1.2)
    lab(ax, "a", x=-0.09)

    # ── b  逐样本：低于对角线 = 输给基线 ──
    ax = axes[1]
    for nm, mm, col in (("HisToGene", M["HisToGene"], P["teal"]),
                        ("Hist2ST", M["Hist2ST"], P["violet"])):
        ids = sorted(set(mm) & set(ref))
        ax.scatter([ref[s][0] for s in ids], [mm[s][0] for s in ids], s=9, alpha=0.75,
                   color=col, edgecolor="none",
                   label=f"{nm} ({sum(1 for s in ids if mm[s][0] > ref[s][0])}/{len(ids)} above)")
        n_s = len(ids)
    lim = [-0.05, 0.85]
    ax.plot(lim, lim, "--", color=P["grey_m"], lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("Ridge + phikon-v2 (PCC)"); ax.set_ylabel("Published method (PCC)")
    ax.set_title(f"Per sample (n={n_s})")
    ax.legend(fontsize=5.5, loc="upper left", handlelength=1.0)
    lab(ax, "b", x=-0.24)
    fig.tight_layout(pad=0.7)
    save(fig, "Fig4_trivial_baseline")


# ══════════════════════════════ Fig 5 ══════════════════════════════
def fig5():
    GR = [16, 8, 4, 2]
    d = {}
    for g in GR:
        for f in glob.glob(f"{RES}/proto_g{g}/*.json"):
            j = jload(f)
            if j:
                d.setdefault(j["name"].replace(f"_g{g}", ""), {})[g] = j
    full = {k: v for k, v in d.items() if all(g in v for g in GR)}
    if not full:
        print("  Fig5 跳过：缺协议扫描"); return

    # Archetype: quantitative grid, hero = panel b（σ 的敏感度才是论点）。
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 56 * MM),
                             gridspec_kw={"width_ratios": [1.0, 1.5, 0.9]})
    pc = {g: [full[k][g]["pcc"] for k in full] for g in GR}
    sg = {g: [full[k][g]["eq_sigma"] for k in full
              if full[k][g].get("eq_flag") == "ok"
              and np.isfinite(full[k][g]["eq_sigma"])] for g in GR}
    xg = np.arange(len(GR))

    for ax, key, dat, ylb, col, ttl in (
            (axes[0], "pcc", pc, "Reported PCC", P["blue"], "PCC barely moves"),
            (axes[1], "eq_sigma", sg, "$\\sigma$ (µm)", P["grey_d"], "$\\sigma$ moves much more")):
        for k in full:
            ax.plot(xg, [full[k][g][key] for g in GR], "-", color=P["grey_l"],
                    lw=0.5, alpha=0.7)
        ax.plot(xg, [np.median(dat[g]) for g in GR], "o-", color=col, lw=1.8, ms=4)
        ax.set_xticks(xg); ax.set_xticklabels([f"{g}×{g}" for g in GR], fontsize=6)
        ax.set_xlabel("Spatial block grid"); ax.set_ylabel(ylb)
        ax.set_title(f"{ttl}\n(n={len(full)} regions)")
    lab(axes[0], "a"); lab(axes[1], "b")

    ax = axes[2]
    rp = 100 * (np.median(pc[16]) - np.median(pc[2])) / np.median(pc[16])
    rs = 100 * (np.median(sg[2]) - np.median(sg[16])) / np.median(sg[16])
    ax.bar([0, 1], [rp, rs], color=[P["blue"], P["grey_d"]], edgecolor=P["grey_d"],
           lw=0.5, width=0.55)
    for i, v in enumerate([rp, rs]):
        ax.text(i, v + rs * 0.03, f"{v:.2f}%", ha="center", fontsize=6.5, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["PCC", "$\\sigma$"], fontsize=7)
    ax.set_ylim(0, rs * 1.18)
    ax.set_ylabel("Relative change, 16×16 → 2×2 (%)")
    ax.set_title(f"$\\sigma$ is {rs/rp:.1f}× more sensitive\nto the split protocol")
    lab(ax, "c")
    fig.tight_layout(pad=0.7)
    save(fig, "Fig5_protocol_sensitivity")


# ══════════════════════════════ Fig 6 ══════════════════════════════
def fig6():
    """六个下游度量。评测栅格固定 16 µm，唯一变量是预测的有效分辨率。

    注意：8 µm 处每格计数不足、受噪声限制，多个度量在 8→16 反而变好；
    单调劣化只在 16→64 µm 成立，标题据实陈述，不写「每一步都劣化」。
    """
    m1, m2 = {}, {}
    for f in glob.glob(f"{RES}/downstream_*.json"):
        d = jload(f)
        if d and all(str(b) in d.get("scales", {}) for b in BINS):
            m1[d["name"]] = d["scales"]
    for f in glob.glob(f"{RES}/downstream2_*.json"):
        d = jload(f)
        if d and all(str(b) in d.get("scales", {}) for b in BINS):
            m2[d["name"]] = d["scales"]
    if not m1:
        print("  Fig6 跳过：缺下游数据"); return
    NSP6 = len(by_specimen(list(m1)))

    METR = [("ari", m1, "Expression-cluster agreement", False, "ARI, predicted vs measured clusters"),
            ("edge_ratio", m1, "Interface gradient fidelity", False, "$|\\nabla$pred$|\\,/\\,|\\nabla$truth$|$"),
            ("coloc_preserve", m2, "Gene co-localisation", False, "Correlation preserved"),
            ("hotspot_jaccard", m2, "Hotspot Jaccard", False, "Jaccard vs truth hotspots"),
            ("boundary_shift_um", m2, "Interface displacement (µm)", True, "Median displacement (µm)")]
    fig, axes = plt.subplots(2, 3, figsize=(183 * MM, 96 * MM))
    axf = axes.ravel()
    drops = []

    for ax, (key, src, ttl, inv, ylb) in zip(axf, METR):
        vals = [[s[str(b)][key] for s in src.values()
                 if isinstance(s[str(b)].get(key), (int, float))
                 and s[str(b)][key] == s[str(b)][key]] for b in BINS]
        med = np.array([np.median(v) if v else np.nan for v in vals])
        col = P["red"] if inv else P["blue"]
        if inv:
            # 该量并未被评测栅格量化——22 个非零位移是 9.2–35.1 µm 的连续值，
            # 没有一个是 16 的整数倍。中位为 0 是因为多数切片的界面**根本没有移动**。
            # 故标注每档「有位移的片数」，而不是谎称存在分辨率下限。
            nz = [sum(1 for x in v if x > 0) for v in vals]
            ax.bar(range(len(BINS)), med, color=col, edgecolor=P["grey_d"],
                   lw=0.5, width=0.6)
            for i, (v, k) in enumerate(zip(med, nz)):
                ax.text(i, (v if v else 0) + max(med) * 0.04,
                        f"{v:.1f}\n{k}/{len(vals[i])}", ha="center", va="bottom",
                        fontsize=5.0, linespacing=1.25,
                        color=P["grey_d"] if v == 0 else P["black"])
            ax.set_ylim(0, max(med) * 1.42)
            ax.text(0.02, 0.97, "median 0 = no shift in most regions;\n"
                                "second line = regions with any shift",
                    transform=ax.transAxes, fontsize=5.0, color=P["grey_m"], va="top")
        else:
            for i, v in enumerate(vals):
                ax.scatter([i] * len(v), v, s=5, color=P["grey_l"], zorder=1)
            ax.plot(range(len(BINS)), med, "o-", color=col, lw=1.5, ms=3.5, zorder=3)
        ax.set_xticks(range(len(BINS))); ax.set_xticklabels(BINS, fontsize=6)
        ax.set_xlabel("Prediction bin size (µm)", fontsize=6)
        ax.set_ylabel(ylb, fontsize=6)
        i16 = BINS.index(16)
        ch = 100 * (med[-1] - med[i16]) / abs(med[i16]) if med[i16] else np.inf
        drops.append(ch)
        ax.set_title(f"{ttl}\n16→64 µm: {med[i16]:.3g} → {med[-1]:.3g}", fontsize=6.2)

    # ── f  按热点尺寸分层的召回：粗输出把「小热点更难」这一结构抹平 ──
    ax = axf[5]
    SZ = ["<50µm", "50-100µm", "100-200µm", "≥200µm"]
    cols = [P["blue"], P["blue2"], P["teal"], P["red"]]
    sel = []
    for b, c in zip(BINS, cols):
        cur = []
        for s in m2.values():
            r = s[str(b)].get("hotspot_recall_by_size") or {}
            cur.append([r[k][0] if k in r else np.nan for k in SZ])
        arr = np.array(cur, float)
        ax.plot(range(len(SZ)), np.nanmedian(arr, 0), "o-", color=c, lw=1.3, ms=3, label=f"{b} µm")
        # 与其余五个度量同口径：逐区域取最大类减最小类，再跨区域取中位（原先是对已取中位的曲线拟合斜率，
        # 那是另一个统计量，results/downstream_summary.json 与正文用的都是这个差值）
        sel.append(float(np.nanmedian(arr[:, -1] - arr[:, 0])))
    ax.set_xticks(range(len(SZ))); ax.set_xticklabels(SZ, rotation=30, ha="right", fontsize=5.5)
    ax.set_xlabel("Hotspot size class", fontsize=6)
    ax.set_ylabel("Recall", fontsize=6)
    ax.set_title("Hotspot recall by size, per-region margin (largest $-$ smallest)\n"
                 f"16→64 µm: {sel[BINS.index(16)]:+.3f} → {sel[-1]:+.3f}",
                 fontsize=6.2)
    ax.legend(fontsize=5.0, ncol=2, handlelength=1.0, loc="upper left",
              title="prediction bin", title_fontsize=5.2)

    for ax, l in zip(axf, "abcdef"):
        lab(ax, l, x=-0.22, y=1.16)
    ndeg = sum(1 for c in drops if c < 0) + (1 if drops[4] > 0 else 0)
    fig.suptitle("Coarsening the prediction degrades every routine spatial readout "
                 "from 16 to 64 µm\n"
                 f"(evaluation grid fixed at 16 µm; n={len(m1)} regions from {NSP6} specimens; "
                 "8 µm is count-noise-limited and not part of the monotone range)",
                 fontsize=7, y=1.03)
    fig.tight_layout(pad=0.8, h_pad=1.6)
    save(fig, "Fig6_downstream_consequences")


if __name__ == "__main__":
    only = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    for k in only:
        print(f"Fig{k} …", flush=True)
        globals()[f"fig{k}"]()
    print("完成")
