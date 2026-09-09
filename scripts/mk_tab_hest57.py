# -*- coding: utf-8 -*-
"""正文用的 57 编码器表（tab_hest57.tex）：两栏并排（29+28 行），列 = 编码器（含引用）、参数量 M、Model PCC、Oracle PCC、Ratio (%)、oracle 领先队列数。
只省去全表的成对差值 ± SE 与 |t| 两列（附录全表保留）。数据直接解析 tab_hestblocks.tex，逐行一致。"""
import re
rows = []
for l in open("paper/tab_hestblocks.tex").read().split("\n"):
    if "&" not in l or l.startswith("\\") or l.startswith("Encoder"): continue
    p = [x.strip() for x in l.rstrip().rstrip("\\\\").split("&")]
    if len(p) < 8: continue
    name, params, model, oracle, ratio, diff, t, coh = p[:8]  # tab_hestblocks.tex 列序：Encoder, Params, Model, Oracle, Ratio, Difference, |t|, Cohorts（2026-09-08 修正：此前 model/oracle 解包反了，正文表两列 PCC 曾互换）
    rows.append((name, params, model, oracle, ratio.replace("\\%", "") + r"\%", coh))  # 比值直接带 % 号
assert len(rows) == 57, len(rows)
k = 29; blocks = [rows[:k], rows[k:]]
# 第二栏末尾的空位放 57 个编码器的平均值行（Model、Oracle、Ratio、领先队列数各取算术平均；参数量不平均）
import numpy as np
mm = np.mean([float(r[2]) for r in rows]); mo = np.mean([float(r[3]) for r in rows]); mr = np.mean([float(r[4].replace("\\%", "")) for r in rows]); mc = np.mean([int(r[5].split("/")[0]) for r in rows])
blocks[1] = blocks[1] + [(r"\textit{Mean over 57}", "--", f"{mm:.4f}", f"{mo:.4f}", f"{mr:.0f}" + r"\%", "--")]  # Coh. 是计数，不取平均
assert len(blocks[1]) == k, len(blocks[1])
hdr = r"Encoder & M & Model & Oracle & Ratio & Coh."
L = [r"\begin{tabular}{@{}lrrrrr@{\hspace{10pt}}|@{\hspace{10pt}}lrrrrr@{}}", r"\toprule", hdr + " & " + hdr + r" \\", r"\midrule"]
for i in range(k):
    cells = []
    for b in blocks:
        cells.append(" & ".join(b[i]) if i < len(b) else " & & & & & ")
    L.append(" & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open("paper/tab_hest57.tex", "w").write("\n".join(L) + "\n"); print(f"tab_hest57.tex: {len(rows)} 行 → 2×{k}，含引用与两个 PCC")
