# -*- coding: utf-8 -*-
"""由 fig1_blocks.py 生成会议版紧凑 Fig 1：去掉两格示意图与 Xenium 那格
（后者移入 Fig 2），重排为 3 行 7 格，高度从 202 mm 压到 118 mm。"""
s = open("fig1_blocks.py").read()
head, _ = s.split("fig = plt.figure(", 1)
plot = r'''fig = plt.figure(figsize=(183 * MM, 118 * MM))
gs = fig.add_gridspec(3, 4, height_ratios=[1.00, 1.00, 0.74],
                      hspace=0.40, wspace=0.10, top=0.885, bottom=0.075,
                      left=0.075, right=0.985)
gb = gs[2, :].subgridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.32)


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)


def letter(ax, ch, x=-0.10, y=1.22):
    ax.text(x, y, ch, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left")


ax = fig.add_subplot(gs[0, 0])
ax.imshow(np.asarray(crop), aspect="equal"); bare(ax)
ax.set_title("H&E", fontsize=8, pad=14)
W_, H_ = crop.size
sb = 200.0 / (DOWN / PX_PER_UM[SLIDE])
ax.plot([W_ * .06, W_ * .06 + sb], [H_ * .945] * 2, "-", color="white", lw=2.6)
ax.text(W_ * .06 + sb / 2, H_ * .915, "200 um", color="white", fontsize=6,
        ha="center", va="bottom")
letter(ax, "a")

R1 = [(FT, "measured", "PC1 projection", None, "b", P["grey_d"]),
      (FD, "blocks only", "truth averaged in 20 image domains",
       map_pcc["dom20"], "c", P["red"]),
      (FP, "trained model", "ridge on frozen features",
       map_pcc["ridge"], "d", P["blue"])]
for i, (F, ttl, sub_, sc, ltr, col) in enumerate(R1):
    ax = fig.add_subplot(gs[0, i + 1])
    ax.imshow(grid(F[sel]), cmap="magma", vmin=vmin, vmax=vmax,
              interpolation="nearest", aspect="equal")
    bare(ax); ax.set_title(ttl, fontsize=8, pad=14)
    ax.text(0.5, 1.045, sub_, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=5.8, color=P["grey_m"])
    if sc is not None:
        ax.text(0.5, -0.04, "map PCC %.3f" % sc, transform=ax.transAxes,
                ha="center", va="top", fontsize=7.4, fontweight="bold", color=col)
    letter(ax, ltr)

gv = np.percentile(np.abs(np.concatenate([GT[sel], GP[sel], GD[sel]])), 99)
ax = fig.add_subplot(gs[1, 0]); ax.axis("off")
ax.text(0.5, 0.58, "finest band\n$B_1 = F - WF$", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=7.4, color=P["grey_d"], linespacing=1.6)
ax.text(0.5, 0.52, "large-scale structure\nremoved", transform=ax.transAxes,
        ha="center", va="top", fontsize=6.4, color=P["grey_m"], linespacing=1.6)
letter(ax, "e", x=-0.10, y=1.08)

R2 = [(GT, None, P["grey_d"]), (GD, map_band["dom20"], P["red"]),
      (GP, map_band["ridge"], P["blue"])]
for i, (G, bp, col) in enumerate(R2):
    ax = fig.add_subplot(gs[1, i + 1])
    ax.imshow(grid(G[sel]), cmap="RdBu_r", vmin=-gv, vmax=gv,
              interpolation="nearest", aspect="equal")
    bare(ax)
    ax.text(0.5, -0.04, "reference" if bp is None else "band PCC %.3f" % bp,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=7.4, fontweight="bold", color=col)

ax = fig.add_subplot(gb[0])
ts = sorted(SIG)
ss = np.array([SIG[t] for t in ts]); pp = np.array([SC["A_coarse_t%d" % t] for t in ts])
ax.plot(ss, pp, "-", color=P["grey_m"], lw=1.4, zorder=1)
ax.plot(ss, pp, "o", ms=3, color=P["grey_m"], mfc="white", mew=1.0, zorder=2)
for k, col, lb, off in (("ridge", P["blue"], "trained model", (-4, 13)),
                        ("dom20", P["red"], "blocks only", (8, 11))):
    ax.plot([sg[k]], [pcc[k]], "o", ms=6.5, color=col, zorder=4)
    ax.annotate("%s\n%.3f - %.0f um" % (lb, pcc[k], sg[k]), (sg[k], pcc[k]),
                textcoords="offset points", xytext=off, fontsize=6.2,
                color=col, fontweight="bold", ha="left")
ax.set_xscale("log")
ax.set_xlabel("equivalent resolution $\\sigma$ (um)", labelpad=1)
ax.set_ylabel("per-gene PCC", labelpad=2)
ax.set_title("the ladder: $\\sigma$ is a monotone function of PCC within a fold",
             fontsize=7.0, pad=5)
ax.text(0.97, 0.95, "grey: truth blurred to $\\sigma$", transform=ax.transAxes,
        ha="right", va="top", fontsize=6.0, color=P["grey_m"])
letter(ax, "f", x=-0.155, y=1.20)

ax = fig.add_subplot(gb[1])
lab_ = ["measured", "blocks\nonly", "trained\nmodel"]
vals = [100 * bpow["truth"], 100 * bpow["dom20"], 100 * bpow["ridge"]]
cols = [P["grey_d"], P["red"], P["blue"]]
ax.bar(np.arange(3), vals, 0.6, color=cols, edgecolor=P["grey_d"], lw=0.4)
for i, v_ in enumerate(vals):
    ax.text(i, v_ + 0.5, "%.1f%%" % v_, ha="center", fontsize=6.4,
            color=cols[i], fontweight="bold")
ax.set_xticks(np.arange(3)); ax.set_xticklabels(lab_, fontsize=6.2)
ax.set_ylabel("variance in the finest band (%)", labelpad=2)
ax.set_ylim(0, max(vals) * 1.30)
ax.set_title("both predictions are %.0f-%.0f$\\times$ too smooth"
             % (bpow["truth"] / bpow["ridge"], bpow["truth"] / bpow["dom20"]),
             fontsize=7.0, pad=5)
letter(ax, "g", x=-0.26, y=1.20)

fig.suptitle("What the score buys is the large-scale pattern, not the fine structure",
             fontsize=9.4, y=0.975, fontweight="bold")
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "Fig1_iclr.pdf"))
print("-> figures/Fig1_iclr.pdf", flush=True)
print("  map PCC %.3f / %.3f ; band %.3f / %.3f"
      % (map_pcc["dom20"], map_pcc["ridge"], map_band["dom20"], map_band["ridge"]), flush=True)
print("  fine-band var share %.1f / %.1f / %.1f%%"
      % (100 * bpow["truth"], 100 * bpow["dom20"], 100 * bpow["ridge"]), flush=True)
'''
open("fig1_iclr.py", "w").write(head + plot)
import ast; ast.parse(head + plot)
print("fig1_iclr.py 已生成")
