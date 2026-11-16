# -*- coding: utf-8 -*-
"""Figure 1, laid out here and finished in Illustrator (svg -> Illustrator -> pdf).
Light rounded panels, icons, short dark labels only; every explanation is in the
caption. a: the OCTAVE framework in one panel. b: the domain oracle and the trained model in the 2 Visium HD windows
of the earlier Figure 1 (full H&E resolution), full map and finest band, with each map's correlation r with the measured map.
Data: results/fig1_framework (panel a) and results/fig1_v3 (panel b), both computed on the cluster.
Writes <out>/Fig1_v4.svg (text kept as text, Arial) and a png preview. Millimetres, y down."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig1_icons as pathology                 # 编码器图标，取自本组绘图素材库
from fig1_icons import P as C

OUT = sys.argv[1] if len(sys.argv) > 1 else "paper/figs"
A = "results/fig1_framework"; B = os.environ.get("FIG1_B", "results/fig1_v3")
ZA = np.load(f"{A}/maps.npz"); NA = json.load(open(f"{A}/numbers.json")); NB = json.load(open(f"{B}/numbers.json"))
plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 6.6,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                     "axes.linewidth": 0.6})
MM = 1 / 25.4
WD, HT = 183.0, 98.0
INK, ARW, EDGE, BG = "#1F2933", "#59636E", "#D6DEE8", "#F4F7FB"
BLUE, RED = "#3F6FA8", "#B64342"
BAND = {1: "#D8645A", 4: "#E39A2D", 7: "#4A7FAD"}
fig = plt.figure(figsize=(WD * MM, HT * MM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, WD); ax.set_ylim(HT, 0); ax.axis("off")
from matplotlib.colors import LinearSegmentedColormap
cmx = LinearSegmentedColormap.from_list("blues_soft", plt.get_cmap("Blues")(np.linspace(0.04, 0.82, 256))); cmx.set_bad("white")
cmb = plt.get_cmap("RdBu_r").copy(); cmb.set_bad("white")


BGL, RAS = [], []                                   # background patches and raster images; everything else is foreground


def box(x, y, w, h, fc, ec, lw=0.6, r=2.2, z=0.5):
    BGL.append(ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc, edgecolor=ec, lw=lw, zorder=z)))


def image(arr, x, y, w, h, r=0.9, ec="#C3CCD7", lw=0.5, z=3, **kw):
    clip = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", transform=ax.transData, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    im = ax.imshow(arr, extent=(x, x + w, y + h, y), interpolation="nearest", zorder=z, **kw); im.set_clip_path(clip); RAS.append(im)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor="none", edgecolor=ec, lw=lw, zorder=z + 0.5))


def arrow(p0, p1, cs=None):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=5.5, lw=0.9, color=ARW, shrinkA=0, shrinkB=0,
                                 zorder=6, connectionstyle=cs or "arc3", joinstyle="miter"))


def label(x, y, s, size=6.6, color=INK, weight="bold", ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=8, **kw)


def letter(x, y, s):
    ax.text(x, y, s, fontsize=9, fontweight="bold", color="#111111", ha="left", va="top", zorder=9)


def lims(maps):
    v = np.concatenate([m[~np.isnan(m)] for m in maps]); return np.percentile(v, [2, 98])

# ============================================================== a: framework
box(4.0, 1.6, 177.5, 45.0, BG, EDGE)
letter(0.4, 2.0, "a")
YT, YM = 14.0, 34.0
he = np.asarray(Image.open(f"{A}/he_window.png").convert("RGB"))
image(he, 9.5, YT - 6.0, 12.0, 12.0, ec="#E3A9C0", lw=0.7)
label(15.5, YT + 8.6, "H&E")
ax.add_patch(Ellipse((15.5, YM), 12.4, 11.4, facecolor="#F7DCE6", edgecolor="#E3A9C0", lw=0.7, zorder=3))
rng = np.random.default_rng(3)
for gx in np.arange(11.0, 20.6, 1.3):
    for gy in np.arange(29.4, 38.8, 1.3):
        if ((gx - 15.5) / 6.0) ** 2 + ((gy - YM) / 5.5) ** 2 < 0.9:
            ax.add_patch(Circle((gx, gy), 0.4, facecolor=cmx(0.35 + 0.6 * rng.random()), edgecolor="none", zorder=4))
label(15.5, YM + 9.9, "Spatial\ntranscriptomics", linespacing=1.1)
for k, (dx, dy, hue) in enumerate(((2.4, -2.0, "s"), (1.2, -1.0, "v"), (0.0, 0.0, "k"))):
    pathology.encoder(ax, 32.6 + dx, YT + dy, 12.5, z=5 + k, hue=hue, frozen=(k == 2))
label(33.4, YT + 8.6, "Frozen encoders")
box(43.8, YT - 3.0, 9.0, 6.0, C["a_pale"], C["a_dark"], lw=0.6, r=1.4, z=5)
label(48.3, YT, "Ridge", size=6.4, color=C["a_dark"])
arrow((22.0, YT), (25.8, YT)); arrow((40.2, YT), (43.5, YT)); arrow((53.1, YT), (56.4, YT)); arrow((22.2, YM), (56.4, YM))
T0, P0 = ZA["T_lev0"], ZA["P_lev0"]; lo, hi = lims([T0, P0])
image(P0, 56.8, YT - 6.0, 12.0, 12.0, cmap=cmx, vmin=lo, vmax=hi)
image(T0, 56.8, YM - 6.0, 12.0, 12.0, cmap=cmx, vmin=lo, vmax=hi)
label(62.8, YT + 8.6, "Prediction Ŷ")
label(62.8, YM + 8.6, "Measurement Y")
cx, cy, ew, eh = 90.4, 24.0, 28.0, 12.0
ax.add_patch(Ellipse((cx, cy + 1.1), ew, eh, facecolor="#C9D5E3", edgecolor="#9FB0C4", lw=0.6, zorder=2))
ax.add_patch(Ellipse((cx, cy), ew, eh, facecolor="#EEF3F9", edgecolor="#9FB0C4", lw=0.6, zorder=2.5))
lat = {}
for i in range(-3, 4):
    for j in range(-1, 2):
        x_, y_ = cx + i * 3.2 + j * 1.2, cy + j * 2.6
        if ((x_ - cx) / (ew / 2 - 1.5)) ** 2 + ((y_ - cy) / (eh / 2 - 1.2)) ** 2 < 1: lat[(i, j)] = (x_, y_)
for (i, j), p in lat.items():
    for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
        q = lat.get((i + di, j + dj))
        if q: ax.plot([p[0], q[0]], [p[1], q[1]], color="#AEBCCD", lw=0.5, zorder=3)
for (i, j), p in lat.items():
    if abs(i) <= 1 and abs(j) <= 1 and (i, j) != (0, 0):
        ax.plot([lat[(0, 0)][0], p[0]], [lat[(0, 0)][1], p[1]], color=C["v_mid"], lw=0.9, zorder=3.2)
for (i, j), p in lat.items():
    hub = (i, j) == (0, 0); nb = abs(i) <= 1 and abs(j) <= 1
    ax.add_patch(Circle(p, 0.75 if hub else 0.55, facecolor=C["v_dark"] if hub else (C["v_light"] if nb else "white"),
                        edgecolor=C["v_dark"] if nb else "#8C9BB0", lw=0.45, zorder=4))
label(cx, cy + 10.4, "Smoothing on the bin graph")
arrow((69.2, YT), (78.0, cy - 2.0), cs="arc3,rad=-0.15"); arrow((69.2, YM), (78.0, cy + 3.0), cs="arc3,rad=0.15")
bx, by, bw, bh = 108.0, 4.4, 38.4, 39.6
box(bx, by, bw, bh, "#EEF4FB", BLUE, lw=0.8, r=2.0, z=1)
label(bx + bw / 2, by + 4.0, "Bands of spatial scale", size=6.8, color=BLUE)
tw, tg = 9.6, 1.4; x0b = bx + 6.0; ts = by + 11.6
for c_, i in enumerate((1, 4, 7)):
    x = x0b + c_ * (tw + tg)
    label(x + tw / 2, by + 8.8, f"Band {i}", size=6.2, color=BAND[i])
    for row, key in enumerate(("P", "T")):
        M = ZA[f"{key}_band{i}"]; g = np.nanpercentile(np.abs(M), 99)
        image(M, x, ts + row * (tw + 1.3), tw, tw, r=0.7, cmap=cmb, vmin=-g, vmax=g, ec=BAND[i], lw=0.7)
    label(x + tw / 2, ts + 2 * tw + 1.3 + 3.0, f"$\\beta_{i}$", size=7.0, color=BAND[i])
label(bx + 3.2, ts + tw / 2, "Ŷ", size=6.6)
label(bx + 3.2, ts + tw * 1.5 + 1.3, "Y", size=6.6)
arrow((cx + ew / 2 + 0.4, cy), (bx - 0.2, cy))
sx, sy, sw, sh = 149.4, 4.4, 31.0, 39.6
box(sx, sy, sw, sh, "#FBF1F0", RED, lw=0.8, r=2.0, z=1)
label(sx + sw / 2, sy + 4.0, "Band scores", size=6.8, color=RED)
pa = fig.add_axes([(sx + 7.6) / WD, 1 - (sy + 31.0) / HT, 20.6 / WD, 21.0 / HT])
sig = NA["sigma_um"]
s_ = np.array([sig[str(2 ** k)] for k in range(12)]); b_ = np.array([NA["beta"][str(2 ** k)] for k in range(12)])
pa.plot(s_, b_, "-", color="#7B8696", lw=1.0, zorder=2); pa.scatter(s_, b_, s=5, color="#7B8696", zorder=3, linewidths=0)
for i in (1, 4, 7): pa.scatter([s_[i - 1]], [b_[i - 1]], s=20, color=BAND[i], zorder=4, edgecolors="white", linewidths=0.5)
pa.set_xscale("log"); pa.set_ylim(0, 1); pa.set_xlim(7, 560)
pa.set_xticks([10, 100]); pa.set_xticklabels(["10", "100"]); pa.set_yticks([0, 0.5, 1]); pa.set_yticklabels(["0", "0.5", "1"])
pa.minorticks_off()
pa.tick_params(labelsize=6.0, length=1.8, width=0.6, pad=1.2, colors=INK)
for sp_ in ("top", "right"): pa.spines[sp_].set_visible(False)
for sp_ in ("left", "bottom"): pa.spines[sp_].set_color(INK)
pa.set_xlabel("Band width (µm)", fontsize=6.3, labelpad=1.0, color=INK); pa.set_ylabel("$\\beta_i$", fontsize=6.8, labelpad=0.5, color=INK)
pa.patch.set_alpha(0)
arrow((bx + bw + 0.2, cy), (sx - 0.2, cy))

# ============================================================== b: the 2 Visium HD windows
py = 48.8
box(4.0, py, 177.5, HT - py - 1.2, BG, EDGE)
letter(0.4, py + 0.4, "b")
tw2, g2 = 14.0, 1.3
hs = 2 * tw2 + 4.6                                   # H&E spans both rows
sec_w = hs + 2.0 + 3 * tw2 + 2 * g2
gap = 6.5
xs0 = 4.0 + (177.5 - 7.0 - (2 * sec_w + gap)) / 2
r1y = py + 10.6; r2y = r1y + tw2 + 4.6
cols = ["Measured", "Domain\noracle", "Trained\nmodel"]
for b_i, TAG in enumerate(("P2", "P5")):
    Z = np.load(f"{B}/maps_{TAG}.npz"); r = NB[TAG]["window_r"]
    xb = xs0 + b_i * (sec_w + gap)
    label(xb + sec_w / 2, py + 3.1, f"Section {b_i + 1}", size=6.8)
    hei = np.asarray(Image.open(f"{B}/he_{TAG}.png").convert("RGB"))
    image(hei, xb, r1y, hs, hs, r=1.1, ec="#E3A9C0", lw=0.7)
    ax.plot([xb + 1.6, xb + 1.6 + 200 / 900 * hs], [r1y + hs - 2.0] * 2, color="white", lw=1.5, solid_capstyle="butt", zorder=6)
    label(xb + hs / 2, r1y - 0.9, "H&E", size=6.4, weight="normal", va="bottom")
    maps = [Z["T_map"], Z["D_map"], Z["P_map"]]; bands = [Z["T_band1"], Z["D_band1"], Z["P_band1"]]
    lo, hi = lims(maps); g = np.nanpercentile(np.abs(np.concatenate([b_[~np.isnan(b_)] for b_ in bands])), 99)
    for c_ in range(3):
        x = xb + hs + 2.0 + c_ * (tw2 + g2)
        label(x + tw2 / 2, r1y - 0.9, cols[c_], size=6.4, weight="normal", va="bottom", linespacing=1.05)
        image(maps[c_], x, r1y, tw2, tw2, cmap=cmx, vmin=lo, vmax=hi)
        image(bands[c_], x, r2y, tw2, tw2, cmap=cmb, vmin=-g, vmax=g)
        if c_ >= 1:
            col, key = (RED, "d") if c_ == 1 else (BLUE, "r")
            label(x + tw2 / 2, r1y + tw2 + 2.1, f"$r$ = {r['map_' + key]:.3f}", size=6.2, color=col)
            label(x + tw2 / 2, r2y + tw2 + 2.1, f"$r$ = {r['band_' + key]:.3f}", size=6.2, color=col)
xr = xs0 + 2 * sec_w + gap + 3.0
label(xr, r1y + tw2 / 2, "Full map", size=6.4, rotation=270)
label(xr, r2y + tw2 / 2, "Finest band", size=6.4, rotation=270)
os.makedirs(OUT, exist_ok=True)
fig.savefig(f"{OUT}/Fig1_v4_preview.png", dpi=300)
ALL = [a for a in ax.get_children() if hasattr(a, "set_visible")] + [pa]
def only(keep):
    for a in ALL: a.set_visible(a in keep)
only(set(BGL)); ax.patch.set_visible(False); fig.patch.set_alpha(0)
fig.savefig(f"{OUT}/Fig1_v4_bg.svg", transparent=True)
only(set(RAS)); fig.savefig(f"{OUT}/Fig1_v4_raster.png", dpi=600, transparent=True)
only(set(ALL) - set(BGL) - set(RAS)); fig.savefig(f"{OUT}/Fig1_v4_fg.svg", transparent=True)
print("ok", HT, {t: {k: round(v, 3) for k, v in NB[t]["window_r"].items()} for t in NB})
