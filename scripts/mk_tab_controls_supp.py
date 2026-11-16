# -*- coding: utf-8 -*-
"""附录的划分对照表（tab_controls_supp.tex）：HEST 与 Xenium 并排，排法同正文表 1。统一汇总规则。
HEST：results/hest_rule_numbers.json 的 summary（每个编码器 72 个样本的中位数，再跨 57 个编码器取中位数；范围为跨编码器；
队列数为每个编码器高于训练模型的队列数的中位数）。Xenium：results/supp57_local.json 的 controls_xen（scripts/supp57_local.py 写出）。
「Of trained model」由印出的 PCC 相除。用法：在项目根目录运行 python3 scripts/mk_tab_controls_supp.py <论文目录>。"""
import json, os, sys

R = os.environ.get("OCTAVE_RESULTS", "results"); PAPER = sys.argv[1] if len(sys.argv) > 1 else "paper"
S = json.load(open(f"{R}/hest_rule_numbers.json"))["summary"]
X = json.load(open(f"{R}/supp57_local.json"))["controls_xen"]
r3 = lambda x: round(x, 3)
ROWS = [("Ridge on frozen features (trained model)", "model", "Ridge on frozen features"),
        ("Image partition, measured means (domain oracle)", "oracle", "Image partition, measured means ($K=20$, domain oracle)"),
        ("Coordinate partition, measured means", "coord", "Coordinate partition, measured means ($K=20$)"),
        ("Random matched-size partition, measured means", "random", "Random matched-size partition, measured means ($K=20$)"),
        ("Image partition, means from training data (learnable domain-level predictor)", "train", "Image partition, means from training blocks ($K=20$)"),
        ("Coordinate partition, means from training data", "train_coord", "Coordinate partition, means from training blocks ($K=20$)"),
        ("Image partition and means both learned from training data", "inductive", "Image partition learned on training blocks, means from training blocks ($K=20$)"),
        None,
        ("Image partition, measured means, $K=200$", None, "Image partition, measured means ($K=200$)"),
        ("Coordinate partition, measured means, $K=200$", None, "Coordinate partition, measured means ($K=200$)")]
hm, xm = r3(S["model_pcc"]["median"]), r3(X["Ridge on frozen features"]["pcc"])
L = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
     r" & \multicolumn{4}{c}{HEST} & \multicolumn{3}{c}{Xenium} \\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
     r"Predictor ($K=20$ unless stated) & PCC & Range over encoders & Of trained model & Cohorts above trained model & PCC & Range over specimens & Of trained model \\",
     r"\midrule"]
for r in ROWS:
    if r is None: L.append(r"\midrule"); continue
    lab, hk, xk = r
    if hk:
        v = S[f"{hk}_pcc"]; lo, hi = v["range"]; h = r3(v["median"])
        coh = "n/a" if hk == "model" else f"{S[hk + '_coh']['median']:.0f}/10"
        hc = f"{h:.3f} & {lo:.3f}--{hi:.3f} & {100 * h / hm:.0f}\\% & {coh}"
    else:
        hc = "n/a & n/a & n/a & n/a"
    x = X[xk]; xc = f"{x['pcc']:.3f} & {x['lo']:.3f}--{x['hi']:.3f} & {100 * r3(x['pcc']) / xm:.0f}\\%"
    L.append(f"{lab} & {hc} & {xc} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{PAPER}/tab_controls_supp.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L))
