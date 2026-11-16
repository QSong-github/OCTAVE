# -*- coding: utf-8 -*-
"""TRIPLEX 对照表（<论文目录>/tab_triplex.tex）。统一汇总规则：TRIPLEX 每折一个分（作者的评测单位），取跨折中位数；
ridge 每样本一个分，取队列内样本的中位数；汇总行取跨 10 个队列的中位数。不用均值。
输入：TRIPLEX 逐折结果 + 同特征的 ridge 逐样本分 + 57 个编码器的 ridge 逐样本分。用法：python3 scripts/mk_tab_triplex.py <论文目录>。
折数取基准自带的划分；某个特征上 TRIPLEX 实际完成的折数少于基准折数时，该格加剑号并在脚注说明。"""
import json, glob, os, sys, numpy as np

R = os.environ.get("OCTAVE_RESULTS", "results"); PAPER = sys.argv[1] if len(sys.argv) > 1 else "paper"
T = json.load(open(os.environ.get("OCTAVE_TRIPLEX", "results_new/triplex/triplex_hest.json")))
COH = {k: v["cohort"] for k, v in json.load(open(f"{R}/hest_blocks_ciga.json"))["samples"].items()}
ORDER = ["SKCM", "HCC", "LUNG", "PAAD", "COAD", "READ", "IDC", "LYMPH_IDC", "PRAD", "CCRCC"]
FEAT = [("cigar", "CIGA (ResNet-18) features"), ("uni_v1", "UNI features")]

def cohort_median(enc):
    ps = json.load(open(f"{R}/hest_rsel_ps_{enc}.json"))["per_sample_pcc"]
    g = {}
    for s, v in ps.items(): g.setdefault(COH[s], []).append(v)
    return {c: float(np.median(v)) for c, v in g.items()}

RID = {e: cohort_median(e) for e in ("ciga", "uni_v1")}
TMED = {k: float(np.median([f["pcc_best_ckpt"] for f in v["folds"].values()])) for k, v in T.items()}   # TRIPLEX：跨折中位数
BEST = {}
for f in sorted(glob.glob(f"{R}/hest_rsel_ps_*.json")):
    for c, v in cohort_median(os.path.basename(f)[len("hest_rsel_ps_"):-5]).items():
        BEST[c] = max(BEST.get(c, -9.0), v)
NENC = len(glob.glob(f"{R}/hest_rsel_ps_*.json"))

nfold = {c: max(T[f"{c}_{k}"]["n_folds"] for k, _ in FEAT) for c in ORDER}
short = [(c, k) for c in ORDER for k, _ in FEAT if T[f"{c}_{k}"]["n_folds"] < nfold[c]]

L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
     r"Cohort & Folds & \multicolumn{2}{c}{CIGA (ResNet-18) features} & \multicolumn{2}{c}{UNI features} & Best of $%d$ \\" % NENC,
     r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}", r" & & TRIPLEX & ridge & TRIPLEX & ridge & ridge \\", r"\midrule"]
for c in ORDER:
    cells = []
    for k, _ in FEAT:
        e = "ciga" if k == "cigar" else k
        v = T[f"{c}_{k}"]
        mark = r"\rlap{$^{\dagger}$}" if v["n_folds"] < nfold[c] else ""
        cells += [f"{TMED[f'{c}_{k}']:.3f}{mark}", f"{RID[e][c]:.3f}"]
    L.append(f"{c.replace('_', chr(92) + '_')} & {nfold[c]} & " + " & ".join(cells) + f" & {BEST[c]:.3f} " + r"\\")
mean = []
for k, _ in FEAT:
    e = "ciga" if k == "cigar" else k
    mean += [np.median([TMED[f"{c}_{k}"] for c in ORDER]), np.median([RID[e][c] for c in ORDER])]
short_bb = {k for c, k in short}
mean_cells = []
for (k, _), i in zip(FEAT, range(0, len(mean), 2)):
    m = r"\rlap{$^{\dagger}$}" if k in short_bb else ""
    mean_cells += [f"{mean[i]:.3f}{m}", f"{mean[i+1]:.3f}"]
L += [r"\midrule", "Median & & " + " & ".join(mean_cells) + f" & {np.median([BEST[c] for c in ORDER]):.3f} " + r"\\",
      r"\bottomrule", r"\end{tabular}"]
open(f"{PAPER}/tab_triplex.tex", "w").write("\n".join(L) + "\n")

note = ""
if short:
    def _one(c, k):
        nf = T[f"{c}_{k}"]["n_folds"]; back = "ResNet-18" if k == "cigar" else "UNI"
        tail = "that fold's value." if nf == 1 else "the median over those folds."
        return (f"On {c.replace('_', chr(92) + '_')} the {back} run completed ${nf}$ of the ${nfold[c]}$ folds, "
                f"and its entry ($\\dagger$) is {tail}")
    note = " ".join(_one(c, k) for c, k in short)
open(f"{PAPER}/triplex_note.tex", "w").write(note + "\n")
print(f"tab_triplex: {len(ORDER)} cohorts, best-of-{NENC} median {np.median([BEST[c] for c in ORDER]):.3f}; 短折 {short}")
for k, e in (("cigar", "ciga"), ("uni_v1", "uni_v1")):
    print(k, "TRIPLEX 高于同特征 ridge 的队列:", sum(round(TMED[f"{c}_{k}"], 3) > round(RID[e][c], 3) for c in ORDER), "/10；高于 57 个编码器中最好者:", [c for c in ORDER if round(TMED[f"{c}_{k}"], 3) > round(BEST[c], 3)])
