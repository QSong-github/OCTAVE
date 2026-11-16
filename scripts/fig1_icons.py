# -*- coding: utf-8 -*-
"""Figure 1 的编码器图标（取自本组的绘图素材库，只保留 fig1_v4.py 用到的部分，绘制结果与原库相同）。
坐标为毫米，y 轴向下。图标函数 f(ax, cx, cy, s, z) 以 (cx, cy) 为中心、边长 s 的正方形内作图；
函数内部用单位坐标（[-0.5, 0.5] 的正方形）书写，再由 U(cx, cy, s) 映射。"""
import math
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle

PT = 72 / 25.4                                              # 每毫米的点数
OUT = 0.45                                                  # 轮廓线宽，pt

P = dict(
    s_dark="#4E5968", s_mid="#7B8696", s_light="#B9C2CD", s_pale="#E3E8EE",          # slate
    a_dark="#C47F22", a_mid="#E9A23B", a_light="#F6CB7C", a_pale="#FCEBCB",          # amber
    k_dark="#4A7FAD", k_mid="#8DB9DE", k_light="#D3E6F5",                            # sky
    v_dark="#6E5590", v_mid="#9A7BB8", v_light="#DCCFEA", v_pale="#F1ECF7",          # violet
)


def U(cx, cy, s):
    """单位坐标 → 毫米。"""
    return lambda x, y: (cx + x * s, cy + y * s)


def tint(c, t):
    return "#%02X%02X%02X" % tuple(round(int(c[i:i + 2], 16) * t + 255 * (1 - t)) for i in (1, 3, 5))


def line(ax, pts, color, w_mm, z=5, cap="round"):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=w_mm * PT, solid_capstyle=cap, solid_joinstyle="round", zorder=z)


def circ(ax, c, r, fc, ec=None, lw=OUT, z=5):
    p = Circle(c, r, facecolor=fc, edgecolor=ec or "none", lw=lw if ec else 0, zorder=z)
    ax.add_patch(p); return p


def path(ax, u, cmds, fc, ec=None, lw=OUT, z=5):
    """cmds 为单位坐标：('M', x, y) ('L', x, y) ('Q', x1, y1, x, y) ('Z',)。"""
    V, C = [], []
    for c in cmds:
        k, a = c[0], c[1:]
        if k == "M":
            V.append(u(*a)); C.append(Path.MOVETO); start = u(*a)
        elif k == "L":
            V.append(u(*a)); C.append(Path.LINETO)
        elif k == "Q":
            V += [u(a[0], a[1]), u(a[2], a[3])]; C += [Path.CURVE3] * 2
        elif k == "Z":
            V.append(start); C.append(Path.CLOSEPOLY)
    p = PathPatch(Path(V, C), facecolor=fc, edgecolor=ec or "none", lw=lw if ec else 0, joinstyle="round", capstyle="round", zorder=z)
    ax.add_patch(p); return p


def ball(ax, c, r, base, light, dark, z=6):
    """节点：底色圆、深色边、小高光。"""
    circ(ax, c, r, base, dark, OUT, z)
    circ(ax, (c[0] - 0.32 * r, c[1] - 0.32 * r), 0.36 * r, light, None, z=z + 0.1)


def encoder(ax, cx, cy, s, z=5, hue="k", frozen=True):
    """预训练编码器：左宽右窄的网络，冻结时右下角加雪花标记。hue 取 P 的色系键（s、k、a、v）。"""
    u = U(cx, cy, s)
    path(ax, u, [("M", -0.42, -0.38), ("L", 0.3, -0.18), ("Q", 0.36, -0.16, 0.36, -0.1), ("L", 0.36, 0.1),
                 ("Q", 0.36, 0.16, 0.3, 0.18), ("L", -0.42, 0.38), ("Q", -0.46, 0.38, -0.46, 0.34), ("L", -0.46, -0.34),
                 ("Q", -0.46, -0.38, -0.42, -0.38), ("Z",)], tint(P[hue + "_light"], 0.7), P[hue + "_dark"], z=z)
    layers = [(-0.33, [-0.24, -0.08, 0.08, 0.24]), (-0.06, [-0.15, 0.0, 0.15]), (0.2, [-0.07, 0.07])]
    for (x0, l0), (x1, l1) in zip(layers, layers[1:]):
        for y0 in l0:
            for y1 in l1:
                line(ax, [u(x0, y0), u(x1, y1)], tint(P[hue + "_dark"], 0.45), 0.014 * s, z=z + 0.1)
    for x, ys in layers:
        for y in ys:
            ball(ax, u(x, y), 0.05 * s, P[hue + "_mid"], P[hue + "_light"], P[hue + "_dark"], z=z + 0.2)
    if frozen:                                                        # 雪花：权重不更新
        circ(ax, u(0.3, -0.32), 0.13 * s, "white", P["k_dark"], z=z + 0.3)
        for k in range(6):
            a = math.radians(30 + 60 * k)
            tip = (0.3 + 0.09 * math.cos(a), -0.32 + 0.09 * math.sin(a))
            line(ax, [u(0.3, -0.32), u(*tip)], P["k_dark"], 0.016 * s, z=z + 0.35)
            for d in (-1, 1):
                b = a + d * math.radians(40)
                mid = (0.3 + 0.055 * math.cos(a), -0.32 + 0.055 * math.sin(a))
                line(ax, [u(*mid), u(mid[0] + 0.03 * math.cos(b), mid[1] + 0.03 * math.sin(b))], P["k_dark"], 0.012 * s, z=z + 0.35)
