# -*- coding: utf-8 -*-
"""正文表 1（tab_controls_main.tex）：域 oracle 与其他划分对训练模型，3 个数据集并排，统一汇总规则。
HEST：results/hest_rule_numbers.json（每个编码器取 72 个样本的中位数，再跨 57 个编码器取中位数）。
Xenium：每个编码器逐区域 → 标本内中位数 → 跨 13 个标本中位数，再跨 57 个编码器取中位数。
  模型与图像 oracle 取自 results/blocks_xen_bands_<编码器>（Hibou-L 为 blocks_xen_bands_base），训练侧均值与随机划分取自 results/xen_ctrl57，
  归纳式取自 results/xen_ind57；坐标划分与编码器无关，取自 results/xen_controls。
Visium HD：两张切片的中位数代替标本，模型与 oracle 取自 results/blocks_vhd_bands_<编码器>，其余取自 results/vhd_ctrl57 与 vhd_ind57。
「of model」由印出的 PCC 相除。用法：在项目根目录运行 python3 scripts/mk_tab_controls_main.py <论文目录>。"""
import json, glob, os, sys, numpy as np

R = os.environ.get("OCTAVE_RESULTS", "results"); PAPER = sys.argv[1] if len(sys.argv) > 1 else "paper"
SPEC = ["Human_Breast_Biomarkers_S1", "Human_Breast_Biomarkers_S2", "Human_Breast_Biomarkers_S3", "Human_Breast_Biomarkers_S4",
        "Xenium_Prime_Cervical", "Xenium_Prime_Ovarian", "Xenium_V1_Human_Kidney", "Xenium_V1_Human_Ovary",
        "Lung", "Xenium_Prime_Breast_Cancer", "Xenium_Prime_Human_Prostate", "Xenium_Prime_Human_Skin", "Xenium_Prime_Human_Lymph_Node"]


def sp(n):
    if "Human_Lung_Cancer_FFPE" in n: return "Lung"   # Xenium v1 与 Prime 5K 为同一供体同一组织块，按一个标本计
    return next(s for s in SPEC if n.startswith(s))


def load(d): return {json.load(open(f))["name"]: json.load(open(f)) for f in glob.glob(f"{R}/{d}/*.json")}


def xmed(vals):                         # Xenium：标本内中位数，再跨标本中位数
    per = {}
    for n, v in vals.items(): per.setdefault(sp(n), []).append(v)
    assert len(per) == 13
    return float(np.median([np.median(v) for v in per.values()]))


ENC = sorted(d.split("/")[-1] for d in glob.glob(f"{R}/xen_ctrl57/*"))
assert len(ENC) == 57
XC = load("xen_controls"); coord = xmed({n: d["K20"]["oracle_coord"]["pcc"] for n, d in XC.items()})
X, V = {k: [] for k in ("ridge", "img", "train", "ind", "rand")}, {k: [] for k in ("ridge", "img", "train", "ind", "coord", "rand")}
for e in ENC:
    B = load("blocks_xen_bands_base" if e == "hibou_l" else f"blocks_xen_bands_{e}"); C = load(f"xen_ctrl57/{e}"); I = load(f"xen_ind57/{e}")
    assert sorted(B) == sorted(C) == sorted(I) == sorted(XC)
    X["ridge"].append(xmed({n: d["pred"]["ridge"]["pcc"] for n, d in B.items()})); X["img"].append(xmed({n: d["pred"]["dom20"]["pcc"] for n, d in B.items()}))
    X["train"].append(xmed({n: d["trainonly_image"] for n, d in C.items()})); X["rand"].append(xmed({n: d["oracle_random_matched"] for n, d in C.items()}))
    X["ind"].append(xmed({n: d["inductive"] for n, d in I.items()}))
    VB = load(f"blocks_vhd_bands_{e}"); VC = load(f"vhd_ctrl57/{e}"); VI = load(f"vhd_ind57/{e}")
    assert len(VB) == len(VC) == len(VI) == 2
    vm = lambda f, D: float(np.median([f(d) for d in D.values()]))   # 两张切片的中位数
    V["ridge"].append(vm(lambda d: d["pred"]["ridge"]["pcc"], VB)); V["img"].append(vm(lambda d: d["pred"]["dom20"]["pcc"], VB))
    V["train"].append(vm(lambda d: d["trainonly_image"], VC)); V["coord"].append(vm(lambda d: d["oracle_coord"], VC))
    V["rand"].append(vm(lambda d: d["oracle_random_matched"], VC)); V["ind"].append(vm(lambda d: d["inductive"], VI))
xm = {k: float(np.median(v)) for k, v in X.items()}; xm["coord"] = coord
vh = {k: float(np.median(v)) for k, v in V.items()}
S = json.load(open(f"{R}/hest_rule_numbers.json"))["summary"]
hm = dict(ridge=S["model_pcc"]["median"], img=S["oracle_pcc"]["median"], train=S["train_pcc"]["median"], ind=S["inductive_pcc"]["median"],
          coord=S["coord_pcc"]["median"], rand=S["random_pcc"]["median"])
ROWS = [("Ridge on frozen features (trained model)", "ridge"), ("Image partition, measured means (domain oracle)", "img"),
        ("Image partition, means learned from training data (learnable domain-level predictor)", "train"),
        ("Image partition and means both learned from training data", "ind"), ("Coordinate partition, measured means", "coord"),
        ("Random matched-size partition, measured means", "rand")]
cell = lambda m, k: f"{m[k]:.3f} & {100 * round(m[k], 3) / round(m['ridge'], 3):.0f}\\%"
L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
     r" & \multicolumn{2}{c}{HEST, $\sim\!100\,\mu$m} & \multicolumn{2}{c}{Xenium, $16\,\mu$m} & \multicolumn{2}{c}{Visium HD, $16\,\mu$m} \\",
     r"Predictor ($K=20$) & PCC & of model & PCC & of model & PCC & of model \\", r"\midrule"]
for lab, k in ROWS: L.append(f"{lab} & {cell(hm, k)} & {cell(xm, k)} & {cell(vh, k)} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
out = "\n".join(L) + "\n"
old = open(f"{PAPER}/tab_controls_main.tex").read() if os.path.exists(f"{PAPER}/tab_controls_main.tex") else ""
open(f"{PAPER}/tab_controls_main.tex", "w").write(out)
print(out); print("与原表逐字相同" if out == old else "与原表不同，请检查 diff")
