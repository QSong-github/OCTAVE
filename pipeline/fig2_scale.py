# -*- coding: utf-8 -*-
"""Fig 2 (ICLR) — 标量掩盖了什么：缺口随尺度的剖面，及其样本层级检验。

a  块 oracle 相对训练模型的缺口随带宽 σ 的变化，每个独立样本一条（区域内取中位）；
   水平虚线是标量缺口的样本中位，注出它落在哪个 σ 上。
b  逐样本配对：标量缺口 vs 最细带缺口，精确双侧符号检验。
c  各带的方差占比：真值 vs 两个预测 —— 两者在细端都远低于真值。

口径：区域先在样本内取中位再做检验（区域计数是伪重复）。
"""
import json, glob, os, re
import numpy as np
from math import comb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("S4ST", "/path/to/project")
OUT = os.path.join(BASE, "figures")
D = os.path.join(BASE, "results/blocks_xen_bands")

P = {"blue": "#0F4D92", "red": "#B64342", "teal": "#42949E", "grey_l": "#CFCECE",
     "grey_m": "#767676", "grey_d": "#4D4D4D", "black": "#272727"}
MM = 1 / 25.4
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.8, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5})


def specimen(n):
    m = re.match(r"(Human_Breast_Biomarkers_S\d)_(Top|Mid|Bot)$", n)
    if m: return m.group(1)
    if "Human_Lung_Cancer_FFPE" in n: return "Lung"   # v1 与 Prime 5K 同一供体同一组织块，按一个标本计
    return n


def signp(k, n):
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


REG = {}
for f in sorted(glob.glob(D + "/*.json")):
    d = json.load(open(f))
    R = d["pred"]
    if "bands" not in R.get("ridge", {}):
        continue
    sig = {int(k): v for k, v in d["sigma_um"].items()}
    ts = sorted(int(t) for t in R["ridge"]["bands"])
    gap = []
    for t in ts:
        r, b = R["ridge"]["bands"][str(t)], R["dom20"]["bands"][str(t)]
        gap.append(100 * (r - b) / r if r > 1e-9 else np.nan)
    REG[d["name"]] = {
        "sigma": np.array([sig[t] for t in ts]),
        "gap": np.array(gap),
        "scalar": 100 * (R["ridge"]["pcc"] - R["dom20"]["pcc"]) / R["ridge"]["pcc"],
        "vt": np.array([R["truth"]["band_var"][str(t)] for t in ts]),
        "vb": np.array([R["dom20"]["band_var"][str(t)] for t in ts]),
        "vr": np.array([R["ridge"]["band_var"][str(t)] for t in ts]),
    }
print(f"区域 {len(REG)}")
if len(REG) < 4:
    raise SystemExit("产出太少，等作业完成")

SP = {}
for n, r in REG.items():
    SP.setdefault(specimen(n), []).append(r)
SIG = np.median(np.vstack([r["sigma"] for r in REG.values()]), axis=0)   # 跨区域中位带宽，与正文/表 12 的 10.3, 14.2, 19.8, 27.4 一致（原来用第一个区域的带宽，轴标注会与正文差 2 µm）
med = lambda key, v: np.nanmedian(np.vstack([x[key] for x in v]), axis=0)
SPEC = {s: {"gap": med("gap", v), "scalar": float(np.median([x["scalar"] for x in v])),
            "vt": med("vt", v), "vb": med("vb", v), "vr": med("vr", v)}
        for s, v in SP.items()}
names = sorted(SPEC)
print(f"独立样本 {len(names)}")

fig = plt.figure(figsize=(183 * MM, 74 * MM))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.15], wspace=0.40,
                      left=0.075, right=0.985, top=0.735, bottom=0.155)


def letter(ax, ch, x=-0.20, y=1.24):
    ax.text(x, y, ch, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left")


# ── a
ax = fig.add_subplot(gs[0])
for s in names:
    ax.plot(SIG, SPEC[s]["gap"], "-", lw=0.9, color=P["grey_m"], alpha=0.75, zorder=2)
gm = np.nanmedian(np.vstack([SPEC[s]["gap"] for s in names]), axis=0)
ax.plot(SIG, gm, "-o", lw=2.2, ms=3.4, color=P["red"], zorder=4)
sm = float(np.median([SPEC[s]["scalar"] for s in names]))
ax.axhline(sm, color=P["blue"], lw=1.4, ls="--", zorder=3)
j = int(np.nanargmin(np.abs(gm - sm)))
ax.plot([SIG[j]], [sm], "o", ms=6, color=P["blue"], zorder=5)
ax.annotate(f"the scalar reports {sm:.0f}%\n= a band at $\\sigma \\approx$ {SIG[j]:.0f} µm",
            (SIG[j], sm), textcoords="offset points", xytext=(10, 16),
            fontsize=6.3, color=P["blue"], fontweight="bold", linespacing=1.4)
ax.annotate(f"finest band: {gm[0]:.0f}%", (SIG[0], gm[0]), textcoords="offset points",
            xytext=(8, -3), fontsize=6.3, color=P["red"], fontweight="bold")
ax.set_xscale("log")
ax.set_xlabel("band width $\\sigma$ (µm)", labelpad=1)
ax.set_ylabel("shortfall of the domain oracle\nrelative to the trained model (%)", labelpad=2)
ax.set_ylim(0, max(100, float(np.nanmax(gm)) * 1.15))
ax.set_title("the shortfall depends on the scale\nat which it is measured",
             fontsize=7.4, pad=6)
k = sum(1 for s in names if SPEC[s]["gap"][0] > SPEC[s]["scalar"])
Pv = signp(k, len(names))
ax.text(0.98, 0.97, f"grey: {len(names)} independent specimens\nred: median\n"
        f"finest band > scalar in {k}/{len(names)} specimens\n(exact $P$ = {Pv:.4f})",
        transform=ax.transAxes, ha="right", va="top", fontsize=5.9,
        color=P["grey_d"], linespacing=1.5)
letter(ax, "a", x=-0.12)

# ── b (formerly c)
ax = fig.add_subplot(gs[1])
for key, col, lab in (("vt", P["grey_d"], "measured"),
                      ("vb", P["red"], "domains only"),
                      ("vr", P["blue"], "trained model")):
    m_ = np.nanmedian(np.vstack([SPEC[s][key] for s in names]), axis=0)
    ax.plot(SIG, 100 * m_, "-o", lw=1.5, ms=3.0, color=col, label=lab)
ax.set_xscale("log")
ax.set_xlabel("band width $\\sigma$ (µm)", labelpad=1)
ax.set_ylabel("share of that field's variance\nin the band (%)", labelpad=2)
vt0 = 100 * float(np.nanmedian([SPEC[s]["vt"][0] for s in names]))
vr0 = 100 * float(np.nanmedian([SPEC[s]["vr"][0] for s in names]))
ax.set_title(f"at the finest band: measurement {vt0:.0f}%,\ntrained model {vr0:.0f}%", fontsize=7.4, pad=6)
ax.legend(fontsize=6.0, loc="upper right", handlelength=1.3)
letter(ax, "b", x=-0.26)

fig.suptitle("What the scalar hides: the same comparison, resolved over scale",
             fontsize=9.2, y=0.985, fontweight="bold")
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "Fig2_scale.pdf"))
print("-> figures/Fig2_scale.pdf")
mo = float(np.median([SPEC[s]["scalar"] for s in names])); mf = float(np.median([SPEC[s]["gap"][0] for s in names]))
print(f"  标量中位 {mo:.1f}%  最细带中位 {mf:.1f}%  比值 {mf/mo:.2f}x")
print(f"  {k}/{len(names)} 样本，P = {Pv:.5f}")
print(f"  标量对应 sigma ~ {SIG[j]:.0f} um（第 {j+1}/{len(SIG)} 档）")
print(f"  细带方差占比 真值 {vt0:.1f}%  模型 {vr0:.1f}%")
json.dump({"n_regions": len(REG), "n_specimens": len(names),
           "sigma_um": SIG.tolist(), "gap_median": gm.tolist(),
           "scalar_median": mo, "finest_median": mf, "k_win": k, "P": Pv,
           "scalar_matches_sigma_um": float(SIG[j]),
           "by_specimen": {s: {"scalar": SPEC[s]["scalar"],
                               "gap": SPEC[s]["gap"].tolist()} for s in names}},
          open(os.path.join(BASE, "results/fig2_scale.json"), "w"), indent=1)
