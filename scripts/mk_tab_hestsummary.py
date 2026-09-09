# -*- coding: utf-8 -*-
"""正文用的 Table 1 摘要（tab_hestsummary.tex）：按参数量分组与蒸馏学生分组的 oracle 比值中位数、范围、oracle 领先的队列数。
输入 results/hest_blocks_summary.json（blk_k20_sel）与 results/encoder_params.json。全表 57 行仍为 tab_hestblocks.tex（附录）。"""
import json, os, numpy as np
R = os.environ.get("S4ST_RESULTS", "results"); S = {e["enc"]: e for e in json.load(open(f"{R}/hest_blocks_summary.json"))}; P = json.load(open(f"{R}/encoder_params.json"))
distilled = {"litevirchow2", "h0_mini", "litefm", "litefm_l", "litefm_s", "distillpath_is16", "distillpath_ks16"}
encs = sorted(S); par = {e: P[e]["params"] / 1e6 for e in encs}
groups = [("All encoders", encs), ("$<100$M parameters", [e for e in encs if par[e] < 100]), ("$100$--$600$M", [e for e in encs if 100 <= par[e] <= 600]), ("$>600$M", [e for e in encs if par[e] > 600]),
          ("Distilled students ($6$--$87$M)", [e for e in encs if e in distilled]), ("Trained from scratch, $<100$M", [e for e in encs if par[e] < 100 and e not in distilled])]
L = [r"\begin{tabular}{lrrrrrr}", r"\toprule", r"Encoders & $n$ & Model PCC & Oracle PCC & Ratio & Range & Cohorts led \\", r"\midrule"]
for lab, g in groups:
    v = np.array([S[e]["blk_k20_sel"]["share"] for e in g]); mod = np.median([S[e]["blk_k20_sel"]["mod"] for e in g]); blk = np.median([S[e]["blk_k20_sel"]["blk"] for e in g]); won = [S[e]["blk_k20_sel"]["won"] for e in g]
    L.append(f"{lab} & {len(g)} & {mod:.3f} & {blk:.3f} & {np.median(v):.0f}\\% & {v.min():.0f}--{v.max():.0f}\\% & {int(np.median(won))}/10 ({sum(w == 10 for w in won)} at $10/10$) \\\\")
L += [r"\bottomrule", r"\end{tabular}"]; open("paper/tab_hestsummary.tex", "w").write("\n".join(L) + "\n"); print("\n".join(L))
