# -*- coding: utf-8 -*-
"""正文用的紧凑分区对照表（tab_controls_main.tex）：基准（队列级中位数再跨编码器取中位）与 Xenium（标本级中位数）并排。输入 results/controls_numbers.json。"""
import json, os
R = os.environ.get("S4ST_RESULTS", "results"); n = json.load(open(f"{R}/controls_numbers.json")); h = n["hest"]; x = n["xen"]
rows = [("Ridge on frozen features (trained model)", h["ridge"]["median"], 1.0, x["ridge"][0], 1.0),
        ("Image partition, measured means (domain oracle)", h["img"]["median"], h["img_over_ridge"]["median"], x["img"][0], x["img_frac"][0]),
        ("Image partition, means learned from training data", h["train"]["median"], h["train_over_ridge"]["median"], x["train_img"][0], x["train_img_frac"][0]),
        ("Image partition and means both learned from training data", h["ind"]["median"], h["ind_over_ridge"]["median"], x["ind_img"][0], x["ind_img_frac"][0]),
        ("Coordinate partition, measured means", h["coord"]["median"], h["coord_over_ridge"]["median"], x["coord"][0], x["coord_frac"][0]),
        ("Random matched-size partition, measured means", h["rand"]["median"], h["rand_over_ridge"]["median"], x["rand"][0], x["rand_frac"][0])]
L = [r"\begin{tabular}{lrrrr}", r"\toprule", r" & \multicolumn{2}{c}{Benchmark, $\sim\!100\,\mu$m} & \multicolumn{2}{c}{Xenium, $16\,\mu$m} \\", r"Predictor ($K=20$) & PCC & of model & PCC & of model \\", r"\midrule"]
for lab, a, fa, b, fb in rows: L.append(f"{lab} & {a:.3f} & {100*fa:.0f}\\% & {b:.3f} & {100*fb:.0f}\\% \\\\")
L += [r"\bottomrule", r"\end{tabular}"]; open("paper/tab_controls_main.tex", "w").write("\n".join(L) + "\n"); print("\n".join(L))
