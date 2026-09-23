#!/usr/bin/env python
"""新增主图。与 make_figs.py 共用样式与调色板（统一调色板）。

FigM「报告的 PCC 到底在测什么」—— 方案里价值最高的一张，此前完全没画。
核心结论：报告分数很大程度上在测**组织本身有多平滑**，而不是模型有多好；
        等效分辨率对同一混杂的依赖显著更弱，且方向相反。

证据链（每格一条，互不重复）：
  a  预测的 Moran's I 逐片被抬高，且压缩到一个与组织无关的固定高值   ← hero
  b  Moran 膨胀随等效 σ 增大 —— σ 的独立佐证（别的数据都给不了这条）
  c  跨片 PCC 追随组织真实 Moran's I，控制噪声天花板后仍成立
  d  片内逐基因同样成立：PCC 是被基因自身的空间自相关买来的
  e  基因层面的解离：PCC 强烈追随 Moran，σ 的依赖弱得多且反向

统计口径：15/16 个切片区域来自 7/8 个独立样本（乳腺 S1–S4 各含 Top/Mid/Bot）。
按区域计数是伪重复，故一律同时给出样本层级计数，显著性只在样本层级施行
（精确双侧符号检验，n=7 时最小可得 P=0.016，n=8 时 0.0078）。
σ 全部为估计量 A（阶梯匹配），不得与估计量 B 的跨度并列。
"""
import glob, json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from math import comb, factorial

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

# conversion_rate.json 的条目名是中文（内部记录用）；投稿图必须英文。
PROTO_EN = {
    "深度线 · 25 个图像编码器（跨片）": "25 image encoders\ncross-section",
    "方法 · 片内块 CV（旧 200 基因基准）": "200-gene benchmark\nwithin-section block CV",
    "方法 · 跨片留一（旧 200 基因基准）": "200-gene benchmark\nleave-one-section-out",
    "多尺度分箱（对照）": "Multi-scale binning\n(control)",
}


def specimen(n):
    m = re.match(r"(Human_Breast_Biomarkers_S\d)_(Top|Mid|Bot)$", n)
    if "Human_Lung_Cancer_FFPE" in n: return "Lung"   # v1 与 Prime 5K 同一供体同一组织块，按一个标本计
    return m.group(1) if m else n


def signp(k, n):
    """精确双侧符号检验。n 小的时候最小可得 P 本身就是硬下界，必须报出来。"""
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def by_spec(names, vals):
    """{样本: 该样本内各区域取值的中位数}"""
    g = {}
    for n, v in zip(names, vals):
        g.setdefault(specimen(n), []).append(v)
    return {k: float(np.median(v)) for k, v in g.items()}


def group_idx(names):
    """{样本: [下标]} —— 需要按样本索引原数组时用它，不要拿 by_spec 的中位数当下标。"""
    g = {}
    for i, n in enumerate(names):
        g.setdefault(specimen(n), []).append(i)
    return g


def lab(ax, s, x=-0.20, y=1.13):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="left", va="bottom")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUT}/{name}.pdf", flush=True)


def jload_(p):
    """缺文件时返回 None —— 让调用方显式跳过，而不是抛异常中断整张图。"""
    try:
        return json.load(open(p))
    except Exception:
        return None


def fig_moran():
    # ── 切片级：16 个区域 ──
    S = []
    for f in sorted(glob.glob(f"{RES}/xenium/*.json")):
        d = json.load(open(f))
        if all(k in d for k in ("moran_true", "moran_pred", "pcc")):
            S.append(d)
    if not S:
        print("  FigM 跳过：缺 xenium 结果"); return
    nm = [d["name"] for d in S]
    mt = np.array([d["moran_true"] for d in S])
    mp = np.array([d["moran_pred"] for d in S])
    pc = np.array([d["pcc"] for d in S])
    sg = np.array([d.get("eq_sigma", np.nan) for d in S])
    ok_s = np.array([d.get("eq_flag") == "ok" for d in S])
    pmr = np.array([d.get("pcc_moran_r", np.nan) for d in S])
    ce = np.array([d.get("ceiling", np.nan) for d in S])
    NSP = len(set(map(specimen, nm)))

    fig = plt.figure(figsize=(183 * MM, 182 * MM))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 0.9, 0.9], hspace=1.20, wspace=1.9)

    # ── a  hero：真值 → 预测 的 Moran's I 配对 ──
    ax = fig.add_subplot(gs[0, :3])
    for a_, b_ in zip(mt, mp):
        ax.plot([0, 1], [a_, b_], "-", color=P["grey_l"], lw=0.7, zorder=1)
    ax.scatter([0] * len(mt), mt, s=16, color=P["grey_d"], marker="o", zorder=3, label="measured")
    ax.scatter([1] * len(mp), mp, s=16, color=P["red"], marker="^", zorder=3, label="predicted")
    ax.plot([0, 1], [np.median(mt), np.median(mp)], "-", color=P["black"], lw=2, zorder=4)
    ax.set_xlim(-0.28, 1.28); ax.set_xticks([0, 1])
    ax.set_xticklabels(["measured", "predicted"], fontsize=6.5)
    ax.set_ylabel("Moran's $I$ of expression")
    up = int((mp > mt).sum())
    ups = sum(1 for k, v in by_spec(nm, (mp - mt)).items() if v > 0)
    ax.set_title("Prediction inflates spatial autocorrelation")
    _ovv = np.array(sorted(by_spec(nm, mp - mt).values()))
    ax.text(0.5, -0.17, f"{up}/{len(mt)} regions, {ups}/{NSP} specimens "
            f"(P = {signp(ups, NSP):.4f});  median gap +{np.median(_ovv):.2f}\n"
            f"specimen range {_ovv.min():.2f}–{_ovv.max():.2f}; "
            f"shape of the gap not resolvable at n = {NSP}",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.4,
            linespacing=1.45, color=P["grey_d"])
    ax.legend(fontsize=5.5, loc="upper left", handlelength=1.0, borderpad=0.1)
    # 内嵌：预测值一律高于真值。**两种形状声称都已撤回**——
    #   「压缩」：OLS 斜率 0.55 = r x sd比 = 0.635 x 0.870，主由回归稀释；
    #             8 样本均值的 Pitman-Morgan 配对方差检验 P = 0.90。
    #   「均匀偏移」：样本层级偏移 0.18–0.51（2.84 倍，CV 0.30），
    #             且随真实 Moran 上升而缩小（r = -0.52）。同样不成立。
    # n=8 只能确立**方向**。故只报中位与全距，并明说形状不可辨。
    # 注：偏移 = pred − true 是逐基因膨胀的**均值**；JSON 里的 moran_diff 是其
    #     **中位**（src/nine.py:156），两者差 ≤0.032，即该分布的偏度。
    ins = ax.inset_axes([0.62, 0.04, 0.35, 0.36])
    ins.scatter(mt, mp, s=7, color=P["red"], zorder=3)
    lo, hi = 0.2, 1.0
    ins.plot([lo, hi], [lo, hi], "--", color=P["grey_m"], lw=0.7)
    off = float(np.median(list(by_spec(nm, mp - mt).values())))
    ins.set_xlim(lo, hi); ins.set_ylim(lo, hi)
    ins.tick_params(labelsize=5.0, length=1.8, pad=1)
    ins.set_xticks([0.4, 0.8]); ins.set_yticks([0.4, 0.8])
    _ov = np.array(sorted(by_spec(nm, mp - mt).values()))
    ins.scatter(list(by_spec(nm, mt).values()), list(by_spec(nm, mp).values()),
                s=20, marker="D", facecolor="none", edgecolor=P["black"], lw=0.8, zorder=4)
    ins.text(0.95, 0.06, "all above identity", transform=ins.transAxes,
             ha="right", va="bottom", fontsize=4.8, color=P["black"])
    ins.set_title("regions $\\bullet$   specimen medians $\\diamond$",
                  fontsize=4.8, pad=2, color=P["grey_d"])
    for sp_ in ins.spines.values():
        sp_.set_linewidth(0.5); sp_.set_color(P["grey_m"])
    for sp_ in ("top", "right"):
        ins.spines[sp_].set_visible(True)
    lab(ax, "a", x=-0.11)

    # ── b  Moran 膨胀 vs 等效 σ：σ 的独立佐证 ──
    ax = fig.add_subplot(gs[0, 3:])
    infl = mp - mt
    m = ok_s & np.isfinite(sg)
    ax.scatter(sg[m], infl[m], s=16, color=P["blue"], edgecolor="white", lw=0.4, zorder=3)
    sp_sig = by_spec(np.array(nm)[m], sg[m]); sp_inf = by_spec(np.array(nm)[m], infl[m])
    ks = sorted(sp_sig)
    ax.scatter([sp_sig[k] for k in ks], [sp_inf[k] for k in ks], s=34, marker="D",
               facecolor="none", edgecolor=P["black"], lw=0.9, zorder=4)
    r_r, r_p = pearsonr(sg[m], infl[m])
    rs_r, _ = pearsonr([sp_sig[k] for k in ks], [sp_inf[k] for k in ks])
    ax.set_xlabel("Effective resolution $\\sigma$ (µm, estimator A)")
    ax.set_ylabel("Moran's $I$ inflation\n(predicted − measured)")
    # 两个方向的声称都不成立：
    #   「独立佐证」错 —— 膨胀与 PCC 共享 76–80% 方差。
    #   「convergent, not independent」也错 —— 控制 PCC 后偏相关仍 +0.83（CI 排除 0），
    #     且 σ 有 47% 的样本层级方差不由 PCC 解释。
    # 此前印的「控制 PCC + 真实 Moran」偏相关 +0.06 是**构造性过度调整**：
    #   膨胀 ≡ pred − true，控制 true 等于按系数 −1 剥掉被解释变量的定义成分；
    #   且该值的置换 P = 0.88、留一法有 3/8 折变号，是噪声。故弃用。
    from numpy.linalg import lstsq
    def _resid(y, X):
        A = np.column_stack([np.ones(len(y))] + X)
        return y - A @ lstsq(A, y, rcond=None)[0]
    _sp_i = np.array(list(by_spec(np.array(nm)[m], infl[m]).values()))
    _sp_s = np.array(list(by_spec(np.array(nm)[m], sg[m]).values()))
    _sp_p = np.array(list(by_spec(np.array(nm)[m], pc[m]).values()))
    _sp_c = np.array(list(by_spec(np.array(nm)[m], ce[m]).values()))
    _pr = pearsonr(_resid(_sp_i, [_sp_p]), _resid(_sp_s, [_sp_p]))[0]
    _pc_ = pearsonr(_resid(_sp_i, [_sp_c]), _resid(_sp_s, [_sp_c]))[0]
    ax.set_title("Inflation tracks effective resolution\n"
                 "beyond what PCC explains")
    ax.text(0.5, -0.30, f"specimen medians $r$ = {rs_r:+.3f} (n = {len(ks)});  partial "
            f"given PCC {_pr:+.3f};  given noise ceiling {_pc_:+.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.4,
            color=P["grey_d"])
    lab(ax, "b", x=-0.14)

    # ── c  跨片：PCC 追随组织真实 Moran's I ──
    ax = fig.add_subplot(gs[1, :3])
    ax.scatter(mt, pc, s=16, color=P["blue"], edgecolor="white", lw=0.4, zorder=3)
    b1c, b0c = np.polyfit(mt, pc, 1)
    xx = np.linspace(mt.min(), mt.max(), 30)
    ax.plot(xx, b0c + b1c * xx, "-", color=P["red"], lw=1.2, zorder=2)
    rc, _ = pearsonr(mt, pc)
    # 区域层级置换是伪重复；P 必须在 8 个样本中位上做精确置换（8! = 40320）
    import itertools
    _sm = by_spec(nm, mt); _sp = by_spec(nm, pc); _kk = sorted(_sm)
    _x = np.array([_sm[k] for k in _kk]); _y = np.array([_sp[k] for k in _kk])
    _obs = abs(pearsonr(_x, _y)[0])
    _cnt = sum(1 for pm in itertools.permutations(range(len(_x)))
               if abs(pearsonr(_x, _y[list(pm)])[0]) >= _obs - 1e-12)
    _pexact = _cnt / factorial(len(_x))
    ax.set_xlabel("Measured Moran's $I$ of the tissue")
    ax.set_ylabel("Reported PCC")
    ax.set_title("Score ranks tissue\nsmoothness", fontsize=6.6)
    ax.text(0.03, 0.97, f"$r$ = {rc:.3f}\n$R^2$ = {rc**2:.3f}\n"
            f"n = {len(mt)} regions\nP = {_pexact:.4f} ({len(_x)} specimens)",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.4, color=P["grey_d"])
    lab(ax, "c")

    # ── d (formerly e)  基因层面的解离：PCC 追随 Moran，σ 弱得多且反向 ──
    ax = fig.add_subplot(gs[1, 3:])
    rows = []
    # 面板 d 的范围与 a–c/e/f 一致：只取 results/xenium 里有的区域（首发 16，Cervical 无 per-gene ⇒ 15）
    _xen_names = {os.path.basename(f) for f in glob.glob(f"{RES}/xenium/*.json")}
    for f in sorted(glob.glob(f"{RES}/per_gene_xen/*.json")):
        if os.path.basename(f) not in _xen_names:
            continue
        d = json.load(open(f)); G = d["genes"]
        pg = np.array([G[g]["pcc"] for g in G]); mg = np.array([G[g]["moran"] for g in G])
        eg = np.array([G[g]["eq"] for g in G])
        mm = np.isfinite(eg) & (eg > 0) & np.isfinite(mg)
        rows.append((d["name"], pearsonr(pg, mg)[0],
                     pearsonr(np.log10(eg[mm]), mg[mm])[0]))
    if rows:
        nmg = [r[0] for r in rows]
        rp = np.array([r[1] for r in rows]); re_ = np.array([r[2] for r in rows])
        for a_, b_ in zip(rp, re_):
            ax.plot([0, 1], [a_, b_], "-", color=P["grey_l"], lw=0.7, zorder=1)
        ax.scatter([0] * len(rp), rp, s=14, color=P["blue"], marker="o", zorder=3)
        ax.scatter([1] * len(re_), re_, s=14, color=P["violet"], marker="s", zorder=3)
        ax.axhline(0, color=P["grey_m"], lw=0.8, ls="--")
        ax.set_xlim(-0.3, 1.3); ax.set_xticks([0, 1])
        ax.set_xticklabels(["per-gene\nPCC", "per-gene\n$\\log\\sigma$"], fontsize=6)
        ax.set_ylabel("$r$ with per-gene Moran's $I$")
        sp_p = by_spec(nmg, rp); sp_e = by_spec(nmg, re_)
        kk = sorted(sp_p)
        kgap = sum(1 for k in kk if abs(sp_p[k]) > abs(sp_e[k]))
        ax.set_title("Confound hits PCC far\nharder than $\\sigma$", fontsize=6.6)
        ax.text(0.5, -0.34, f"median {np.median(list(sp_p.values())):+.2f} vs "
                f"{np.median(list(sp_e.values())):+.2f};  {kgap}/{len(kk)} specimens "
                f"(P = {signp(kgap, len(kk)):.4f})", transform=ax.transAxes,
                ha="center", va="top", fontsize=5.6, color=P["grey_d"])
    lab(ax, "d", x=-0.16)

    # ── f  达成的分数 vs 拆半噪声天花板：目标不是 1.0 ──
    ax = fig.add_subplot(gs[2, :3])
    o_ = np.argsort(ce)
    yy = np.arange(len(o_))
    for i_, k_ in enumerate(o_):
        ax.plot([pc[k_], ce[k_]], [i_, i_], "-", color=P["grey_l"], lw=1.4, zorder=1)
    ax.scatter(pc[o_], yy, s=13, color=P["blue"], marker="o", zorder=3, label="achieved PCC")
    ax.scatter(ce[o_], yy, s=13, color=P["grey_d"], marker="|", lw=1.3, zorder=3,
               label="split-half ceiling")
    ax.set_yticks([]); ax.set_ylabel(f"{len(S)} regions", fontsize=6)
    ax.set_xlabel("PCC", fontsize=6, labelpad=1)
    _kc = sum(1 for v in by_spec(nm, ce - pc).values() if v > 0)
    ax.set_title(f"The attainable target is {np.median(ce):.2f}, not 1.00", fontsize=6.4)
    ax.legend(fontsize=5.0, loc="upper left", handlelength=1.2, borderpad=0.2)
    ax.text(0.5, -0.30, f"ceiling exceeds achieved PCC in {int((ce>pc).sum())}/{len(S)} "
            f"regions and {_kc}/{NSP} specimens (P = {signp(_kc, NSP):.4f})",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.2,
            color=P["grey_d"])
    lab(ax, "e", x=-0.10, y=1.14)

    # ── g  归一化掉噪声后，名次几乎不动 ──
    ax = fig.add_subplot(gs[2, 3:])
    sk_ = pc / ce
    rp_ = np.argsort(np.argsort(-pc)); rs_ = np.argsort(np.argsort(-sk_))
    for i_ in range(len(pc)):
        ax.plot([0, 1], [rp_[i_], rs_[i_]], "-", color=P["grey_l"], lw=0.7, zorder=1)
    ax.scatter([0] * len(pc), rp_, s=12, color=P["blue"], zorder=3)
    ax.scatter([1] * len(pc), rs_, s=12, facecolor="none", edgecolor=P["grey_d"],
               lw=0.8, zorder=3)
    ax.set_xlim(-0.3, 1.3); ax.set_xticks([0, 1])
    ax.set_xticklabels(["raw PCC", "PCC / ceiling"], fontsize=6)
    ax.set_ylabel("rank (0 = best)", fontsize=6); ax.invert_yaxis()
    _fr = (1 - ce) / (1 - pc)
    ax.set_title("Dividing out the noise floor\nbarely reorders anything", fontsize=6.4)
    ax.text(0.5, -0.22, f"Spearman $r_s$ = {spearmanr(pc, sk_).statistic:.3f}; "
            f"max shift {int(np.abs(rp_-rs_).max())} places\n"
            f"irreducible noise is {100*np.median(_fr):.0f}% of the remaining gap "
            f"({100*_fr.min():.0f}–{100*_fr.max():.0f}%)",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.2,
            linespacing=1.4, color=P["grey_d"])
    lab(ax, "f", x=-0.16, y=1.14)

    save(fig, "Fig2_what_pcc_measures")


# ══════════════════════════ Fig 1（schematic-led composite）══════════════════════════
def fig1_intro():
    """Fig 1 = 背景/框架/方法/数据集 + 度量的三段论。

    archetype 改为 schematic-led composite：
    上半是两格示意图（任务与度量的定义），下半是四格定量证据。
    示意图不由 matplotlib 画 —— 留出精确占位框，另附生成提示词，
    最终在 Illustrator 里合成。占位框的长宽比即成图应满足的长宽比。
    """
    d = json.load(open(f"{RES}/effres_hibou_l_hvg50_t2048.json"))
    sig = {int(k): v for k, v in d["sigma_um"].items()}
    cps = sorted(sig); sv = np.array([sig[c] for c in cps])
    tp = {int(k): v for k, v in d["truth_power"].items()}
    meths = sorted(d["methods"], key=lambda m: -m["pcc"])
    TAU = "0.2"

    # 示意图区加高：b 要装「模糊阶梯 + 匹配曲线」两行，30 mm 塞不下。
    fig = plt.figure(figsize=(183 * MM, 196 * MM))
    gs = fig.add_gridspec(3, 4, height_ratios=[2.05, 0.95, 1.00],
                          hspace=0.92, wspace=0.72)

    # ── a / b：示意图占位 ──
    for i, (sl, letter, ttl, note) in enumerate((
            (gs[0, :2], "a", "The task and what the field reports",
             "H&E tile → spatial expression;\nevaluation collapses to one scalar PCC"),
            (gs[0, 2:], "b", "Effective resolution, defined",
             "band-pass diffusion curve →\nthe blur σ (µm) that matches the score"))):
        ax = fig.add_subplot(sl)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(True); s_.set_linestyle((0, (4, 4)))
            s_.set_linewidth(0.7); s_.set_color(P["grey_m"])
        ax.text(0.5, 0.55, "schematic", ha="center", va="center",
                fontsize=8, color=P["grey_m"], transform=ax.transAxes)
        ax.text(0.5, 0.40, note, ha="center", va="center",
                fontsize=5.6, color=P["grey_l"], transform=ax.transAxes)
        ax.set_title(ttl)
        lab(ax, letter, x=-0.055, y=1.05)

    # ── c：数据集总览（三条数据线，全部真实计数）──
    ax = fig.add_subplot(gs[1, :2])
    L = json.load(open(f"{RES}/hest_ladder.json"))
    coh, plat = {}, {}
    for v in L.values():
        coh.setdefault(v["cohort"], {"Visium": 0, "Xenium": 0})[v["platform"]] += 1
        plat[v["platform"]] = plat.get(v["platform"], 0) + 1
    order = sorted(coh, key=lambda c: -sum(coh[c].values()))
    y = np.arange(len(order))[::-1]
    vv = np.array([coh[c]["Visium"] for c in order], float)
    xx = np.array([coh[c]["Xenium"] for c in order], float)
    ax.barh(y, vv, color=P["blue"], edgecolor=P["grey_d"], lw=0.4, height=0.68,
            label=f"Visium (n={plat.get('Visium', 0)})")
    ax.barh(y, xx, left=vv, color=P["teal"], edgecolor=P["grey_d"], lw=0.4,
            height=0.68, hatch="///", label=f"Xenium (n={plat.get('Xenium', 0)})")
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=5.6)
    ax.set_xlabel("samples")
    ax.set_title(f"Breadth line: HEST benchmark\n{len(L)} samples, "
                 f"{len(order)} cohorts, 2 platforms", fontsize=6.6)
    ax.legend(fontsize=5.2, loc="lower right", handlelength=1.0)
    lab(ax, "c", x=-0.34, y=1.20)

    # ── d：另两条数据线的规模（Xenium 深度线 / Visium HD 编码器线）──
    ax = fig.add_subplot(gs[1, 2:])
    X = [json.load(open(f)) for f in sorted(glob.glob(f"{RES}/xenium/*.json"))]
    nb = np.array([d_["n"] for d_ in X], float) / 1e3
    o = np.argsort(nb)
    cols = [P["violet"] if specimen(X[i]["name"]).startswith("Human_Breast") else P["red"]
            for i in o]
    ax.barh(np.arange(len(o)), nb[o], color=cols, edgecolor=P["grey_d"],
            lw=0.4, height=0.75)
    ax.set_yticks([]); ax.set_xlabel("bins per region (×10³)")
    nsp = len({specimen(d_["name"]) for d_ in X})
    ax.set_ylabel(f"{len(X)} regions")
    ax.set_title(f"Depth line: Xenium\n{len(X)} regions from {nsp} specimens",
                 fontsize=6.6)
    ax.text(0.97, 0.06, "purple: 4 breast blocks\n(3 regions each)\nred: 4 single-region "
            "specimens", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.0, color=P["grey_d"])
    lab(ax, "d", x=-0.16, y=1.20)

    # ── e：σ 的实测标定（原 Fig1a）──
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(cps, sv, "o-", color=P["blue"], lw=1.2, ms=3)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Diffusion steps $t$"); ax.set_ylabel("Measured $\\sigma$ (µm)")
    sl_ = np.polyfit(np.log(cps), np.log(sv), 1)[0]
    ax.set_title("Calibrated by\nsecond moment", fontsize=6.4)
    ax.text(0.05, 0.95, f"$\\sigma \\propto t^{{{sl_:.2f}}}$", transform=ax.transAxes,
            ha="left", va="top", fontsize=6, color=P["grey_d"])
    lab(ax, "e", x=-0.50, y=1.20)

    # ── f：带通相关曲线 + τ 交点（原 Fig1b，此处为定量 hero）──
    ax = fig.add_subplot(gs[2, 1:3])
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
    ax.set_title(f"Effective resolution at $\\tau$ = {TAU} spans "
                 f"{er.min():.1f}–{er.max():.1f} µm "
                 f"({er.max()/er.min():.2f}×, n = {len(meths)})", fontsize=6.4)
    ax.legend(fontsize=5.2, loc="lower right", handlelength=1.2)
    ins = ax.inset_axes([0.07, 0.56, 0.40, 0.40])
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
    for s_ in ("top", "right"):
        ins.spines[s_].set_visible(True)
    for s_ in ins.spines.values():
        s_.set_linewidth(0.5); s_.set_color(P["grey_m"])
    lab(ax, "f", x=-0.09, y=1.20)

    # ── g：功率的尺度分布 vs 该尺度的保真度（原 Fig1c）──
    ax = fig.add_subplot(gs[2, 3])
    tot = sum(tp.values())
    sh = np.array([100 * tp[c] / tot for c in cps])
    x = np.arange(len(cps))
    ax.bar(x, sh, color=P["grey_l"], edgecolor=P["grey_d"], lw=0.5, width=0.75, zorder=2)
    ax.set_xticks(x[::2])
    ax.set_xticklabels([f"{s:.0f}" for s in sv[::2]], rotation=45, ha="right", fontsize=5.2)
    ax.set_xlabel("Band $\\sigma$ (µm)", fontsize=6)
    ax.set_ylabel("Share of band-passed\ntruth power (%)", fontsize=6)
    best = np.array([{int(k): v for k, v in meths[0]["curve"].items()}[c] for c in cps])
    ax2 = ax.twinx()
    ax2.plot(x, best, "o-", color=P["blue"], lw=1.3, ms=2.5, zorder=3)
    ax2.set_ylabel("Band-wise PCC", color=P["blue"], fontsize=6)
    ax2.tick_params(axis="y", colors=P["blue"], labelsize=5.5)
    ax2.spines["top"].set_visible(False); ax2.set_ylim(0, 1)
    ax.set_title("Signal sits where\nfidelity is lowest", fontsize=6.4)
    ax.text(0.97, 0.30, f"{sh[0]:.0f}% of signal\nat PCC {best[0]:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.4,
            color=P["grey_d"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.6))
    lab(ax, "g", x=-0.55, y=1.20)

    fig.canvas.draw()
    for letter, axp in (("a", fig.axes[0]), ("b", fig.axes[1])):
        bb = axp.get_position()
        w_mm = bb.width * 183.0; h_mm = bb.height * 168.0
        print(f"  占位框 {letter}: {w_mm:.1f} x {h_mm:.1f} mm | 长宽比 "
              f"{w_mm/h_mm:.2f}:1 | 600dpi = {w_mm/25.4*600:.0f} x {h_mm/25.4*600:.0f} px",
              flush=True)
    # 2026-08-19: 本图已被 fig1_blocks.py 的动机图取代，五格内容移入 S14。
    # 保留函数以便复现历史版本，但改名输出，避免覆盖或与新 Fig 1 混淆。
    save(fig, "Fig1_intro_SUPERSEDED")





# ══════════════════════ Fig 6：未报告的协议旋钮 ══════════════════════
def fig_knobs():
    """三个从不被报告、却比换模型影响更大的协议旋钮。

    ⚠ 本图经一轮对抗核查后重写，撤回两处：
      1. 原 panel d 的编码器参照用了 **HEST 15 编码器**跨度 0.1093，却标成
         「all 25 encoders」。同协议（同 2 片、同 hvg50、同 Ridge_HEST）的
         25 编码器跨度是 **0.1160**。改正后只有**一个**旋钮超过它，不是两个；
         归一化属于另一个队列，已从该轴移除。
      2. 原 panel e 的「上下文落在编码器前沿上」是**估计量构造出来的恒等式**：
         σ 是该片阶梯的逆映射，任何只降低 PCC 而不动靶标的旋钮都必然落在同一条
         曲线上（逐片重拟合的比值 0.986–1.000）。已整格替换为真正的机制量——
         **靶标漂移**：基因面板把尺子本身改了，上下文没有。
    """
    import re as _re
    # ── 归一化（HEST 队列，与下面的扫描不同队列，不可同轴比较）──
    enc = sorted(os.path.basename(f)[len("hest_cp10k_"):-5]
                 for f in glob.glob(f"{RES}/hest_cp10k_*.json"))
    _A = {e: json.load(open(f"{RES}/hest_reported_pcc_{e}.json")) for e in enc}
    _B = {e: json.load(open(f"{RES}/hest_cp10k_{e}.json"))["per_sample_pcc"] for e in enc}
    ids0 = sorted(set(k for k, v in _A[enc[0]].items()
                      if isinstance(v, dict) and "pcc" in v) & set(_B[enc[0]]))
    X0 = np.array([np.mean([_A[e][i]["pcc"] for i in ids0]) for e in enc])
    Y0 = np.array([np.mean([_B[e][i] for i in ids0]) for e in enc])
    coh = {i: _A[enc[0]][i]["cohort"] for i in ids0}
    dcoh = {c: float(np.mean([np.mean([_B[e][i] - _A[e][i]["pcc"]
                                       for i in ids0 if coh[i] == c]) for e in enc]))
            for c in sorted(set(coh.values()))}

    # ── 两个扫描（Visium HD P2/P5，单塔 hibou_l）──
    LADCAP = 410.23     # 该轮阶梯的最粗档；σ 超过它即右删失

    def sweep(pat, key, extra=None):
        out = []
        for f in sorted(glob.glob(pat)) + ([extra] if extra else []):
            n = os.path.basename(f)[:-5]
            if n.endswith("g3"):
                continue                     # 同时改了 grid，非单变量
            m_ = _re.search(key + r"(\d+)", f if key == "hvg" else n)
            v = int(m_.group(1)) if m_ else 224
            d = json.load(open(f)); sm = d["_summary"]
            sl = [k for k in d if not k.startswith("_")]
            # 逐片把该片自己的 PCC 匹配到该片自己的阶梯，并标出右删失。
            # 汇总值混了口径（分子单片、分母双片均值），比值不可直接引用。
            per = {}
            for s_ in sl:
                sg_ = {int(k): vv for k, vv in d[s_]["sigma_um"].items()}
                ld_ = {int(k): d[s_]["ladder"][k]["pcc"] for k in d[s_]["ladder"]}
                cps_ = sorted(sg_); pts = [(sg_[c], ld_[c]) for c in cps_]
                pv = d[s_]["methods"]["Ridge_HEST"]["pcc"]
                if pv >= pts[0][1]:
                    per[s_] = (pts[0][0], "left")
                else:
                    per[s_] = (pts[-1][0], "right")
                    for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
                        if v0 >= pv >= v1:
                            w_ = (v0 - pv) / max(v0 - v1, 1e-12)
                            per[s_] = (float(np.exp(np.log(s0) + w_ *
                                       (np.log(s1) - np.log(s0)))), "ok")
                            break
            out.append(dict(x=v, pcc=sm["pcc"]["Ridge_HEST"],
                            sig=sm["eq_sigma"]["Ridge_HEST"]["pcc"], per=per,
                            moran=float(np.mean([d[s]["moran_true"] for s in sl])),
                            lad=float(np.mean([d[s]["ladder"]["8"]["pcc"] for s in sl]))))
        return sorted(out, key=lambda r: r["x"])
    CTX = sweep(f"{RES}/ctx_sweep/hibou_l_ctx*.json", "ctx",
                f"{RES}/tower_sweep/hibou_l.json")
    PAN = sweep(f"{RES}/panel_sweep_hvg*/hibou_l.json", "hvg")

    # 同协议的 25 编码器跨度 —— panel d 唯一合法的参照
    TW = []
    for f in sorted(glob.glob(f"{RES}/tower_sweep/*.json")):
        sm = json.load(open(f)).get("_summary", {})
        if "pcc" in sm and "Ridge_HEST" in sm["pcc"]:
            TW.append(sm["pcc"]["Ridge_HEST"])
    SPAN = max(TW) - min(TW)

    fig = plt.figure(figsize=(183 * MM, 124 * MM))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0], hspace=1.18, wspace=2.0)

    # ── a  归一化：15 个编码器全部下降 ──
    ax = fig.add_subplot(gs[0, :2])
    for x_, y_ in zip(X0, Y0):
        ax.plot([0, 1], [x_, y_], "-", color=P["grey_l"], lw=0.7, zorder=1)
    ax.scatter([0] * len(X0), X0, s=13, color=P["blue"], marker="o", zorder=3)
    ax.scatter([1] * len(Y0), Y0, s=13, color=P["red"], marker="v", zorder=3)
    ax.plot([0, 1], [X0.mean(), Y0.mean()], "-", color=P["black"], lw=2, zorder=4)
    ax.set_xlim(-0.3, 1.3); ax.set_xticks([0, 1])
    ax.set_xticklabels(["log1p\n(as scored)", "CP10K"], fontsize=5.8)
    ax.set_ylabel("Mean per-gene PCC")
    ax.set_title(f"Normalisation choice\ncosts {100*(X0.mean()-Y0.mean())/X0.mean():.0f}% of the score", fontsize=6.4)
    ax.text(0.5, -0.34, f"{int((Y0 < X0).sum())}/{len(X0)} encoders and "
            f"{sum(1 for v in dcoh.values() if v < 0)}/{len(dcoh)} cohorts fall;\n"
            f"mean {X0.mean():.4f} → {Y0.mean():.4f} "
            f"({100*(Y0.mean()-X0.mean())/X0.mean():+.1f}%)  ·  HEST benchmark, 72-sample mean",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.2,
            linespacing=1.4, color=P["grey_d"])
    lab(ax, "a", x=-0.42, y=1.16)

    # ── b  但它不改名次 ──
    ax = fig.add_subplot(gs[0, 2:4])
    rx = len(X0) - 1 - np.argsort(np.argsort(X0))
    ry = len(Y0) - 1 - np.argsort(np.argsort(Y0))
    for i in range(len(rx)):
        ax.plot([0, 1], [rx[i], ry[i]], "-", color=P["grey_l"], lw=0.7, zorder=1)
    ax.scatter([0] * len(rx), rx, s=11, color=P["blue"], zorder=3)
    ax.scatter([1] * len(ry), ry, s=11, color=P["red"], zorder=3)
    conc = sum(1 for i in range(len(X0)) for j in range(i + 1, len(X0))
               if (X0[i] - X0[j]) * (Y0[i] - Y0[j]) > 0)
    npair = len(X0) * (len(X0) - 1) // 2
    tau = (conc - (npair - conc)) / npair
    ax.set_xlim(-0.3, 1.3); ax.set_xticks([0, 1])
    ax.set_xticklabels(["log1p", "CP10K"], fontsize=6)
    ax.set_ylabel("Leaderboard rank"); ax.invert_yaxis()
    ax.set_title("It moves the level,\nnot the order", fontsize=6.4)
    ax.text(0.5, -0.30, f"Spearman $r_s$ = {spearmanr(X0, Y0).statistic:.3f};  "
            f"Kendall $\\tau$ = {tau:.3f}\n{npair-conc}/{npair} discordant pairs;  "
            f"max rank shift {int(np.abs(rx-ry).max())}",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.2,
            linespacing=1.4, color=P["grey_d"])
    lab(ax, "b", x=-0.38, y=1.16)

    # ── c  两个旋钮的 PCC↔σ 轨迹 ──
    ax = fig.add_subplot(gs[0, 4:])
    for v, col, mk, nm_ in ((CTX, P["violet"], "o", "encoder patch"),
                            (PAN, P["teal"], "s", "gene panel")):
        ax.plot([r["pcc"] for r in v], [r["sig"] for r in v], mk + "-", color=col,
                lw=1.2, ms=3.5, label=nm_)
    ax.set_yscale("log")
    ax.set_xlabel("Reported PCC"); ax.set_ylabel("Effective $\\sigma$ (µm)")
    def ratio(v):
        """逐片比值；两端都未删失才算「确定」，否则只作下界。"""
        ok, lo = [], []
        for s_ in v[0]["per"]:
            a_, fa = v[0]["per"][s_]; b_, fb = v[-1]["per"][s_]
            (ok if fa == "ok" and fb == "ok" else lo).append(b_ / a_)
        return ok, lo
    cok, clo = ratio(CTX); pok, plo = ratio(PAN)
    ax.set_title("Same score change,\nvery different blur", fontsize=6.4)
    _ct = (f"crop {np.mean(cok):.1f}× (n = {len(cok)} uncensored"
           + (f"; {len(clo)} censored, $\\geq${np.mean(clo):.1f}×)" if clo else ")"))
    _pt = (f"panel {min(pok):.2f}–{max(pok):.2f}× (n = {len(pok)}, none censored)"
           if pok else "panel: all censored")
    ax.text(0.5, -0.34, _ct + "\n" + _pt, transform=ax.transAxes,
            ha="center", va="top", fontsize=5.2, linespacing=1.4, color=P["grey_d"])
    ax.legend(fontsize=5.2, loc="upper right", handlelength=1.2)
    lab(ax, "c", x=-0.42, y=1.16)

    # ── d  |ΔPCC| vs **同协议** 25 编码器跨度 ──
    ax = fig.add_subplot(gs[1, :2])
    dp = max(r["pcc"] for r in PAN) - min(r["pcc"] for r in PAN)
    dc = max(r["pcc"] for r in CTX) - min(r["pcc"] for r in CTX)
    ax.barh([2, 1, 0], [dp, dc, SPAN],
            color=[P["teal"], P["violet"], P["grey_m"]],
            edgecolor=P["grey_d"], lw=0.4, height=0.6)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["gene panel\n20→200 HVG", "encoder patch\n224→1344 px",
                        f"all {len(TW)} encoders\n(best−worst)"], fontsize=5.4)
    ax.set_xlabel("|ΔPCC|")
    ax.axvline(SPAN, color=P["grey_d"], lw=0.8, ls="--", zorder=0)
    for yy, vv in zip([2, 1, 0], [dp, dc, SPAN]):
        ax.text(vv + 0.005, yy, f"{vv:.3f}", va="center", fontsize=5.6, color=P["grey_d"])
    ax.set_xlim(0, max(dp, SPAN) * 1.30)
    ax.set_title(f"One protocol choice beats\nthe encoder spread ({dp/SPAN:.2f}×)",
                 fontsize=6.4)
    ax.text(0.5, -0.30, "reference refit on the same 2 sections,\n"
            "same 50-HVG target and estimator",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.0,
            linespacing=1.4, color=P["grey_m"])
    lab(ax, "d", x=-0.62, y=1.16)

    # ── e  靶标漂移：一个旋钮改分数，另一个改尺子本身 ──
    ax = fig.add_subplot(gs[1, 2:4])
    xs = np.arange(2); w = 0.34
    dmo = [max(r["moran"] for r in v) - min(r["moran"] for r in v) for v in (PAN, CTX)]
    dla = [max(r["lad"] for r in v) - min(r["lad"] for r in v) for v in (PAN, CTX)]
    ax.bar(xs - w/2, dmo, w, color=P["blue"], edgecolor=P["grey_d"], lw=0.4,
           label="target Moran's $I$")
    ax.bar(xs + w/2, dla, w, color=P["blue2"], edgecolor=P["grey_d"], lw=0.4,
           hatch="///", label="blur calibration")
    ax.set_xticks(xs); ax.set_xticklabels(["gene panel", "encoder patch"], fontsize=6)
    ax.set_ylabel("|change| in the reference")
    for i, (a_, b_) in enumerate(zip(dmo, dla)):
        ax.text(i - w/2, a_ + 0.006, f"{a_:.3f}", ha="center", fontsize=5.4,
                color=P["grey_d"])
        ax.text(i + w/2, b_ + 0.006, f"{b_:.3f}", ha="center", fontsize=5.4,
                color=P["grey_d"])
    ax.set_ylim(0, max(dmo + dla) * 1.30)
    ax.set_title("The gene panel changes the target;\nthe encoder patch does not", fontsize=6.4)
    ax.legend(fontsize=5.0, loc="upper right", handlelength=1.1)
    ax.text(0.5, -0.30, "the crop sweep leaves the target and its calibration curve\n"
            "numerically unchanged at every setting",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.2,
            linespacing=1.4, color=P["grey_d"])
    lab(ax, "e", x=-0.42, y=1.16)

    # ── f  阴性对照：缓冲区几乎不花分数 ──
    ax = fig.add_subplot(gs[1, 4:])
    B = json.load(open(f"{RES}/buffer_cv.json"))
    DS = [0, 25, 50, 100, 200, 400, 800]
    worst = 0.0
    for g, col, mk in ((4, P["blue"], "o"), (8, P["blue2"], "s"), (16, P["teal"], "^")):
        per = []
        for s_ in B:
            c = np.array([B[s_][f"g{g}_d{d_}"]["pcc"] for d_ in DS])
            per.append(100 * (c / c[0] - 1))
            worst = max(worst, abs(per[-1][-1]))
        ax.plot(DS, np.mean(per, 0), mk + "-", color=col, lw=1.1, ms=3,
                label=f"{g}×{g} blocks")
    ax.axhline(0, color=P["grey_m"], lw=0.8)
    ax.set_xlabel("Exclusion buffer (µm)"); ax.set_ylabel("PCC change (%)")
    ax.set_title("Control: spatial exclusion\nbuffer costs little", fontsize=6.4)
    ax.legend(fontsize=5.0, loc="lower left", handlelength=1.2)
    ax.text(0.97, 0.95, f"n = 2 sections; worst single\ncell {worst:.1f}%; curve has not\n"
            "saturated by 800 µm",
            transform=ax.transAxes, ha="right", va="top", fontsize=4.8,
            linespacing=1.35, color=P["grey_m"])
    lab(ax, "f", x=-0.42, y=1.16)

    save(fig, "Fig5_protocol_knobs")


# ══════════════ Fig 3：评测协议决定结论（旧 Fig3 + 旧 Fig5 + 换算率）══════════════
def fig_protocol():
    """把「评测栅格」「划分几何」「换算率」三条线并进一张图 —— 它们回答同一个问题：
    在预测完全不变的前提下，评测设计能把结论改成什么样。

    a–c  评测栅格跟随 vs 固定（15 区域 / 7 样本）
    d–f  空间块划分从 16×16 扫到 2×2（16 区域）
    g–h  换算率：陡峭，且随划分几何而变（≠ 随基准而变，见 §33.3）
    """
    BINS_ = [8, 16, 32, 64]
    pair = []
    for f in sorted(glob.glob(f"{RES}/downstream_*.json")):
        n = os.path.basename(f)[len("downstream_"):-5]
        d = jload_(f)
        if not d or not all(str(b) in d.get("scales", {}) for b in BINS_):
            continue
        fx = [d["scales"][str(b)]["pcc"] for b in BINS_]
        ow = []
        for b in BINS_:
            q = jload_(f"{RES}/xenium_multi/{n}_bin{b}.json" if b != 16
                       else f"{RES}/xenium/{n}.json")
            ow.append(q["pcc"] if q else np.nan)
        if np.isfinite(ow).all():
            pair.append((n, np.array(ow), np.array(fx)))
    N = len(pair)
    SPEC = group_idx([p[0] for p in pair]); NSP = len(SPEC)

    GR = [16, 8, 4, 2]
    pr = {}
    for g in GR:
        for f in glob.glob(f"{RES}/proto_g{g}/*.json"):
            k = jload_(f)
            if k:
                pr.setdefault(k["name"].replace(f"_g{g}", ""), {})[g] = k
    full = {k: v for k, v in pr.items() if all(g in v for g in GR)}

    fig = plt.figure(figsize=(183 * MM, 150 * MM))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 0.95, 0.95], hspace=1.15, wspace=2.35)

    # ── a  同一批预测，两种评测栅格 ──
    ax = fig.add_subplot(gs[0, 1:5])
    for _, ow, fx in pair:
        ax.plot(BINS_, ow, "-", color=P["red"], lw=0.6, alpha=0.4)
        ax.plot(BINS_, fx, "-", color=P["blue"], lw=0.6, alpha=0.4)
    ax.plot(BINS_, np.median([p[1] for p in pair], 0), "o-", color=P["red"], lw=2, ms=4,
            label="grid follows prediction")
    ax.plot(BINS_, np.median([p[2] for p in pair], 0), "s-", color=P["blue"], lw=2, ms=4,
            label="grid fixed at 16 µm")
    ax.set_xscale("log", base=2); ax.set_xticks(BINS_); ax.set_xticklabels(BINS_)
    ax.set_xlabel("Prediction bin size (µm)"); ax.set_ylabel("Reported PCC")
    ax.set_title(f"Identical predictions, opposite conclusion "
                 f"({N} regions / {NSP} specimens)", fontsize=6.6)
    ax.legend(fontsize=5.4, loc="lower left", handlelength=1.4)
    lab(ax, "a", x=-0.09, y=1.20)

    # ── d/e  划分几何扫描 ──
    xg = np.arange(len(GR))
    for sl, key, ylb, col, ttl, letter in (
            (gs[1, :3], "pcc", "Reported PCC", P["blue"], "PCC barely moves", "b"),
            (gs[1, 3:], "eq_sigma", "$\\sigma$ (µm)", P["grey_d"],
             "$\\sigma$ moves much more", "c")):
        ax = fig.add_subplot(sl)
        for k in full:
            ax.plot(xg, [full[k][g][key] for g in GR], "-", color=P["grey_l"],
                    lw=0.5, alpha=0.7)
        vals = {g: [full[k][g][key] for k in full
                    if key == "pcc" or (full[k][g].get("eq_flag") == "ok"
                                        and np.isfinite(full[k][g][key]))] for g in GR}
        ax.plot(xg, [np.median(vals[g]) for g in GR], "o-", color=col, lw=1.8, ms=4)
        ax.set_xticks(xg); ax.set_xticklabels([f"{g}×{g}" for g in GR], fontsize=5.8)
        ax.set_xlabel("Spatial block grid", fontsize=6); ax.set_ylabel(ylb, fontsize=6)
        ax.set_title(f"{ttl}\n(n = {len(full)} regions)", fontsize=6.4)
        lab(ax, letter, x=-0.16, y=1.18)
        if letter == "b":
            rp_ = 100 * (np.median(vals[16]) - np.median(vals[2])) / np.median(vals[16])
        else:
            rs_ = 100 * (np.median(vals[2]) - np.median(vals[16])) / np.median(vals[16])

    # ── d (formerly g)  25 个编码器的 PCC↔σ 换算率 ──
    ax = fig.add_subplot(gs[2, :3])
    rows = []
    for f in sorted(glob.glob(f"{RES}/legacy8k/tower_*.json")):
        n = os.path.basename(f)[len("tower_"):-5]
        if any(x in n for x in ("grid", "ctx")):
            continue
        d = jload_(f)
        if not d:
            continue
        sl_ = [k for k in d if k.startswith("Visium")]
        sg_ = [d[k]["eq_sigma"] for k in sl_ if d[k].get("flag") == "ok"]
        pc_ = [d[k]["pcc"] for k in sl_ if d[k].get("flag") == "ok"]
        if sg_:
            rows.append((n, float(np.mean(pc_)), float(np.mean(sg_))))
    pc = np.array([r[1] for r in rows]); sg = np.array([r[2] for r in rows])
    ax.scatter(pc, sg, s=14, color=P["blue"], edgecolor="white", lw=0.4, zorder=3)
    bb, aa = np.polyfit(pc, np.log(sg), 1)
    xx = np.linspace(pc.min(), pc.max(), 40)
    ax.plot(xx, np.exp(aa + bb * xx), "-", color=P["red"], lw=1.2, zorder=2)
    r2 = 1 - ((np.log(sg) - (aa + bb * pc)) ** 2).sum() / \
             ((np.log(sg) - np.log(sg).mean()) ** 2).sum()
    per = (np.exp(0.01 * bb) - 1) * 100
    ax.set_yscale("log"); ax.set_xlabel("Reported PCC", fontsize=6)
    ax.set_ylabel("Effective $\\sigma$ (µm)", fontsize=6)
    ax.set_title("Steep score-to-blur\nslope", fontsize=6.4)
    ax.text(0.97, 0.95, f"+0.01 PCC $\\Rightarrow$ {per:+.1f}% $\\sigma$\n"
            f"$R^2$ = {r2:.3f},  n = {len(rows)}", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.4, color=P["grey_d"])
    lab(ax, "d", x=-0.16, y=1.18)

    # ── e (formerly h)  换算率随划分几何而变 ──
    ax = fig.add_subplot(gs[2, 3:])
    conv = jload_(f"{RES}/conversion_rate.json") or []
    cs = [c for c in conv if isinstance(c.get("pct_per_0.01pcc"), (int, float))]
    cs = sorted(cs, key=lambda c: c["pct_per_0.01pcc"])
    names = [PROTO_EN.get(c["name"], c["name"]) for c in cs]
    vals = np.array([c["pct_per_0.01pcc"] for c in cs])
    y = np.arange(len(cs))[::-1]
    ax.barh(y, vals, color=P["blue"], edgecolor=P["grey_d"], lw=0.5, height=0.6)
    ax.axvline(0, color=P["grey_m"], lw=0.8)
    ax.set_yticks([])
    ax.set_xlabel("$\\sigma$ change per +0.01 PCC (%)", fontsize=6)
    ax.set_xlim(vals.min() * 1.32, abs(vals.min()) * 0.92)
    for yi, v, nmm in zip(y, vals, names):
        ax.text(v - 0.4, yi, f"{v:+.1f}", va="center", ha="right", fontsize=5.4,
                color=P["grey_d"])
        ax.text(0.5, yi, nmm.replace("\n", " · "), va="center", ha="left",
                fontsize=4.8, color=P["grey_d"])
    ax.set_title(f"Rate spans {vals.min():.1f} to {vals.max():.1f}%\n"
                 "within one benchmark", fontsize=6.4)
    lab(ax, "e", x=-0.16, y=1.18)

    save(fig, "Fig3_evaluation_protocol")


# ═════════ Fig 4：基准分得开样本，分不开方法 ═════════
def fig_methods():
    """把「已发表方法 vs 冻结特征基线」扩成完整的一张图。

    a  逐癌种：三个已发表方法 vs 冻结 phikon-v2 + 岭回归
    b  逐样本：低于对角线即输给基线
    c  BLEEP vs 15 条冻结特征流水线的配对差
    d  方差分解 —— 「你拿到哪个样本」比「你用哪个方法」重要 20 倍
    e  队列层级 120 组两两比较：Holm 校正后一组都过不了（**结构上不可能**）
    f  诚实的反面：PCC 排序与 σ 排序并不打架
    """
    COH_ = ["SKCM", "HCC", "LUNG", "PAAD", "COAD", "READ", "IDC", "LYMPH_IDC",
            "PRAD", "CCRCC"]
    ENC = sorted(os.path.basename(f)[len("hest_reported_pcc_"):-5]
                 for f in glob.glob(f"{RES}/hest_reported_pcc_*.json"))
    ENC = [e for e in ENC if e]
    D = {}
    for e in ENC:
        d = jload_(f"{RES}/hest_reported_pcc_{e}.json") or {}
        D[e] = {k: v["pcc"] for k, v in d.items()
                if isinstance(v, dict) and "pcc" in v}
    ref0 = jload_(f"{RES}/hest_reported_pcc_phikon_v2.json") or {}
    coh = {k: v["cohort"] for k, v in ref0.items()
           if isinstance(v, dict) and "cohort" in v}
    ids = sorted(set.intersection(*[set(D[e]) for e in ENC]))

    def best(pat):
        out = {}
        for f in glob.glob(pat):
            for sid, v in (jload_(f) or {}).items():
                if isinstance(v, dict) and isinstance(v.get("pcc"), (int, float)) \
                        and v["pcc"] == v["pcc"]:
                    if sid not in out or v["pcc"] > out[sid]:
                        out[sid] = v["pcc"]
        return out
    PUB = {"HisToGene": best(f"{RES}/histogene_*.json"),
           "Hist2ST": best(f"{RES}/h2st2_[mp]*.json"),
           "BLEEP": D.get("bleep", {})}
    BASE = D["phikon_v2"]

    fig = plt.figure(figsize=(183 * MM, 178 * MM))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 1.0], hspace=1.15, wspace=2.3)

    # ── a  逐癌种 ──
    ax = fig.add_subplot(gs[0, :3])
    x = np.arange(len(COH_)); w = 0.2
    HAT = ["", "///", "...", ""]
    series = [("HisToGene", PUB["HisToGene"], P["teal"]),
              ("Hist2ST", PUB["Hist2ST"], P["violet"]),
              ("BLEEP", PUB["BLEEP"], P["red"]),
              ("Ridge + phikon-v2", BASE, P["blue"])]
    tab = {}
    for i, (nm, mm, col) in enumerate(series):
        v = [np.mean([mm[s] for s in ids if coh.get(s) == c and s in mm])
             if any(coh.get(s) == c and s in mm for s in ids) else np.nan
             for c in COH_]
        tab[nm] = v
        ax.bar(x + (i - 1.5) * w, v, width=w, color=col, edgecolor=P["grey_d"],
               lw=0.35, label=nm, hatch=HAT[i])
    nwin = sum(1 for j in range(len(COH_))
               if max(tab[k][j] for k in list(tab)[:3]) > tab["Ridge + phikon-v2"][j])
    ax.set_xticks(x); ax.set_xticklabels(COH_, rotation=40, ha="right", fontsize=5.4)
    ax.set_ylabel("Per-gene PCC", fontsize=6)
    lo_ = min(np.nanmin(v) for v in tab.values())
    ax.set_ylim(min(0, lo_ * 1.5) - 0.01, max(np.nanmax(v) for v in tab.values()) * 1.32)
    ax.axhline(0, color=P["grey_m"], lw=0.6, zorder=0)
    ax.set_title(f"Three end-to-end methods vs a linear baseline on frozen features\n"
                 f"(baseline higher in {len(COH_)-nwin}/{len(COH_)} cohorts)",
                 fontsize=6.4)
    ax.legend(fontsize=5.2, ncol=4, loc="upper center", handlelength=1.1,
              columnspacing=1.0, bbox_to_anchor=(0.5, 1.0))
    lab(ax, "a", x=-0.08, y=1.18)

    # ── b  逐样本 ──
    ax = fig.add_subplot(gs[0, 3:])
    for nm, col, mk in (("HisToGene", P["teal"], "o"), ("Hist2ST", P["violet"], "s"),
                        ("BLEEP", P["red"], "^")):
        mm = PUB[nm]; sel = sorted(set(mm) & set(BASE))
        ax.scatter([BASE[s] for s in sel], [mm[s] for s in sel], s=8, alpha=0.7,
                   color=col, marker=mk, edgecolor="none",
                   label=f"{nm} ({sum(1 for s in sel if mm[s] > BASE[s])}/{len(sel)})")
    lim = [-0.08, 0.85]
    ax.plot(lim, lim, "--", color=P["grey_m"], lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("Ridge + phikon-v2 (PCC)", fontsize=6)
    ax.set_ylabel("End-to-end method (PCC)", fontsize=6)
    ax.set_title("Per sample; above diagonal = beats baseline", fontsize=6.4)
    ax.legend(fontsize=5.0, loc="upper left", handlelength=1.0, title="samples won",
              title_fontsize=5.0)
    lab(ax, "b", x=-0.16, y=1.18)

    # ── c  BLEEP vs 15 条冻结特征流水线 ──
    ax = fig.add_subplot(gs[1, :3])
    bl = np.array([D["bleep"][i] for i in ids])
    rows = []
    for e in ENC:
        if e == "bleep":
            continue
        v = np.array([D[e][i] for i in ids])
        cw = sum(1 for c in COH_
                 if np.mean([v[j] for j, i in enumerate(ids) if coh.get(i) == c])
                 > np.mean([bl[j] for j, i in enumerate(ids) if coh.get(i) == c]))
        rows.append((e, v.mean() - bl.mean(), cw))
    rows.sort(key=lambda z: z[1])
    yy = np.arange(len(rows))
    cols = [P["blue"] if r[2] == len(COH_) else P["grey_l"] for r in rows]
    ax.barh(yy, [r[1] for r in rows], color=cols, edgecolor=P["grey_d"], lw=0.35,
            height=0.7)
    ax.axvline(0, color=P["grey_m"], lw=0.8)
    ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=4.4)
    ax.set_xlabel("mean PCC − BLEEP", fontsize=6)
    nall = sum(1 for r in rows if r[2] == len(COH_))
    ax.set_title(f"Frozen features beat BLEEP\n({nall}/{len(rows)} win all "
                 f"{len(COH_)} cohorts)", fontsize=6.4)
    lab(ax, "c", x=-0.42, y=1.18)

    # ── d  方差分解 ──
    ax = fig.add_subplot(gs[1, 3:])
    M = np.array([[D[e][i] for i in ids] for e in ENC])
    gm = M.mean(); sst = ((M - gm) ** 2).sum()
    ssm = len(ids) * ((M.mean(1) - gm) ** 2).sum()
    sss = len(ENC) * ((M.mean(0) - gm) ** 2).sum()
    parts = [100 * sss / sst, 100 * (sst - ssm - sss) / sst, 100 * ssm / sst]
    labs = ["which sample\nyou were handed", "residual", "which method\nyou used"]
    yy_ = [2, 1, 0]
    cols = [P["red"], P["grey_m"], P["blue"]]
    ax.barh(yy_, parts, color=cols, edgecolor=P["grey_d"], lw=0.4, height=0.6)
    for y_, v_, c_ in zip(yy_, parts, cols):
        ax.text(v_ + 1.5, y_, f"{v_:.1f}%", va="center", ha="left", fontsize=6,
                fontweight="bold", color=c_)
    ax.set_yticks(yy_); ax.set_yticklabels(labs, fontsize=5.2, linespacing=1.3)
    ax.set_xlim(0, 112)
    ax.set_xlabel("share of total PCC variance (%)", fontsize=6)
    ax.set_title(f"Sample identity matters {sss/ssm:.0f}× more\n"
                 f"than method identity", fontsize=6.4)
    ax.text(0.5, -0.42, f"{len(ENC)} pipelines × {len(ids)} samples",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.2,
            color=P["grey_d"])
    lab(ax, "d", x=-0.40, y=1.18)

    # ── e  队列层级两两比较：Holm 后全军覆没 ──
    ax = fig.add_subplot(gs[2, :3])
    CM = np.array([[np.mean([D[e][i] for i in ids if coh.get(i) == c]) for c in COH_]
                   for e in ENC])
    ps = []
    for a_ in range(len(ENC)):
        for b_ in range(a_ + 1, len(ENC)):
            k = int((CM[a_] > CM[b_]).sum())
            ps.append(signp(max(k, len(COH_) - k), len(COH_)))
    ps = np.array(sorted(ps))
    holm = 0.05 / np.arange(len(ps), 0, -1)
    floor = signp(len(COH_), len(COH_))
    ax.plot(np.arange(1, len(ps) + 1), ps, "-", color=P["blue"], lw=1.3,
            label="observed P (sorted)")
    ax.plot(np.arange(1, len(ps) + 1), holm, "--", color=P["red"], lw=1.1,
            label="Holm threshold")
    ax.axhline(floor, color=P["grey_d"], lw=0.9, ls=":",
               label=f"attainable floor ({floor:.5f})")
    ax.set_yscale("log")
    ax.set_xlabel(f"pairwise comparison (of {len(ps)})", fontsize=6)
    ax.set_ylabel("P (exact sign test)", fontsize=6)
    ax.set_title(f"{int((ps<0.05).sum())}/{len(ps)} significant uncorrected,\n"
                 f"{int((ps<=holm).sum())}/{len(ps)} after Holm", fontsize=6.4)
    ax.legend(fontsize=4.8, loc="upper left", handlelength=1.3, borderpad=0.2)
    ax.text(0.5, -0.30, "the attainable floor lies ABOVE the Holm threshold:\n"
            "no pair can pass, whatever the data",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.0,
            color=P["grey_d"], linespacing=1.35)
    lab(ax, "e", x=-0.11, y=1.18)

    # ── f  诚实的反面：PCC 排序与 σ 排序并不打架 ──
    #     没有这一格，读者会把全文读成「PCC 会误导方法选择」，而数据否定这一点。
    ax = fig.add_subplot(gs[2, 3:])
    for tag, ff, col, mk, off in (
            ("cross-section LOO", "method_rank_cross", P["blue"], "o", 0.0),
            ("within-section block CV", "method_rank_within", P["teal"], "s", 1.7)):
        d_ = jload_(f"{RES}/{ff}.json")
        if not d_:
            continue
        rr = d_["rows"]
        pcc_ = np.array([r[1] for r in rr]); sig_ = np.array([r[2] for r in rr])
        rp_ = np.argsort(np.argsort(-pcc_))     # PCC 越高名次越前
        rs_ = np.argsort(np.argsort(sig_))      # σ 越小名次越前
        for i_ in range(len(rr)):
            ax.plot([off, off + 1], [rp_[i_], rs_[i_]], "-", color=P["grey_l"],
                    lw=0.7, zorder=1)
        ax.scatter([off] * len(rr), rp_, s=13, color=col, marker=mk, zorder=3)
        ax.scatter([off + 1] * len(rr), rs_, s=13, facecolor="none", edgecolor=col,
                   lw=0.8, marker=mk, zorder=3)
        ax.text((off + 0.5) / 3.7 + 0.135, -0.20,
                f"{tag}\n$\\rho$ = {d_['spearman']:.3f}  (n = {len(rr)})",
                transform=ax.transAxes, ha="center", va="top", fontsize=5.0,
                color=col, linespacing=1.35)
    ax.set_xticks([0, 1, 1.7, 2.7])
    ax.set_xticklabels(["PCC", "$\\sigma$", "PCC", "$\\sigma$"], fontsize=6)
    ax.set_ylabel("rank (0 = best)", fontsize=6)
    ax.invert_yaxis(); ax.set_xlim(-0.5, 3.2)
    ax.set_title("The honest negative: PCC rank and\n$\\sigma$ rank do not disagree",
                 fontsize=6.4)
    lab(ax, "f", x=-0.14, y=1.18)

    save(fig, "Fig4_what_benchmarks_resolve")


if __name__ == "__main__":
    which = sys.argv[1:] or ["M", "1", "6", "3", "4"]
    if "M" in which:
        print("FigM …", flush=True); fig_moran()
    if "1" in which:
        print("Fig1 …", flush=True); fig1_intro()
    if "6" in which:
        print("Fig6 …", flush=True); fig_knobs()
    if "3" in which:
        print("Fig3 …", flush=True); fig_protocol()
    if "4" in which:
        print("Fig4 …", flush=True); fig_methods()
    print("完成")
