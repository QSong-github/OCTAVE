# -*- coding: utf-8 -*-
"""按统一汇总规则写 HEST 的表（规则见 scripts/hest_rule_numbers.py：每个数先逐样本得到，比值与差值也先逐样本算，
再队列内中位数、跨 10 个队列中位数、跨 57 个编码器中位数）。
写出 tab_hestblocks.tex、tab_hestsummary.tex、tab_zscore.tex、tab_controls_hest.tex、tab_methods.tex，并把 tab_triplex.tex 的汇总行改成中位数。
输入 results/hest_rule_numbers.json。
用法：在项目根目录运行 python3 scripts/mk_tab_hest_rule.py <论文目录>。"""
import json, os, sys, numpy as np

R = os.environ.get("OCTAVE_RESULTS", "results"); PAPER = sys.argv[1] if len(sys.argv) > 1 else "paper"
J = json.load(open(f"{R}/hest_rule_numbers.json")); E = {o["enc"]: o for o in J["encoders"]}; S = J["summary"]
pc = lambda x: f"{100 * x:.0f}\\%"
med = lambda k: float(np.median([E[e][k] for e in E]))
rd = lambda f: open(f"{PAPER}/{f}").read()
wr = lambda f, L: open(f"{PAPER}/{f}", "w").write("\n".join(L) + "\n")

# 编码器在表中的显示名与引用键，参数量取自 encoder_params
LABEL = {"biomedclip": r"BiomedCLIP \citep{biomedclip}",
         "ciga": r"CIGA \citep{ciga}",
         "clip_vitl14": r"CLIP-L/14 \citep{clip}",
         "conch_v1": r"CONCH v1 \citep{conch}",
         "conch_v15": r"CONCH v1.5 \citep{titan}",
         "ctranspath": r"CTransPath \citep{ctranspath}",
         "dinov2_base": r"DINOv2-B \citep{dinov2}",
         "dinov2_giant": r"DINOv2-g \citep{dinov2}",
         "dinov2_large": r"DINOv2-L \citep{dinov2}",
         "dinov3_vitb16": r"DINOv3-B \citep{dinov3}",
         "dinov3_vith16": r"DINOv3-H+ \citep{dinov3}",
         "dinov3_vitl16": r"DINOv3-L \citep{dinov3}",
         "distillpath_is16": r"DistillPath-IS16 \citep{distillpath}",
         "distillpath_ks16": r"DistillPath-KS16 \citep{distillpath}",
         "genbio_pathfm": r"GenBio-PathFM \citep{genbiopathfm}",
         "gigapath": r"Prov-GigaPath \citep{gigapath}",
         "gigapath_flash": r"GigaPath-Flash \citep{gigapathflash}",
         "gpfm": r"GPFM \citep{gpfm}",
         "h0_mini": r"H0-mini \citep{h0mini}",
         "hibou_b": r"Hibou-B \citep{hibou}",
         "hibou_l": r"Hibou-L \citep{hibou}",
         "hoptimus0": r"H-optimus-0 \citep{hoptimus}",
         "hoptimus1": r"H-optimus-1 \citep{hoptimus1}",
         "kaiko_vitb16": r"Kaiko-B16 \citep{kaiko}",
         "kaiko_vitb8": r"Kaiko-B8 \citep{kaiko}",
         "kaiko_vitl14": r"Kaiko-L14 \citep{kaiko}",
         "kaiko_vits16": r"Kaiko-S16 \citep{kaiko}",
         "keep": r"KEEP \citep{keep}",
         "litefm": r"LiteFM \citep{litepath}",
         "litefm_l": r"LiteFM-L \citep{litepath}",
         "litefm_s": r"LiteFM-S \citep{litepath}",
         "litevirchow2": r"LiteVirchow2 \citep{litepath}",
         "lunit_r50_bt": r"Lunit-R50-BT \citep{lunit}",
         "lunit_r50_moco": r"Lunit-R50-MoCo \citep{lunit}",
         "lunit_r50_swav": r"Lunit-R50-SwAV \citep{lunit}",
         "lunit_vits16": r"Lunit-S16 \citep{lunit}",
         "lunit_vits8": r"Lunit-S8 \citep{lunit}",
         "mascaret": r"Mascaret \citep{waivrobust}",
         "midnight12k": r"Midnight-12k \citep{midnight}",
         "mstar": r"mSTAR \citep{mstar}",
         "musk": r"MUSK \citep{musk}",
         "omiclip": r"OmiCLIP \citep{omiclip}",
         "openmidnight": r"OpenMidnight \citep{openmidnight}",
         "path_foundation": r"Path Foundation \citep{pathfoundation}",
         "pathgen_clip": r"PathGen-CLIP \citep{pathgen}",
         "pathryoshka_b": r"Pathryoshka-B \citep{pathryoshka}",
         "phaet": r"Phaet \citep{waivrobust}",
         "phikon": r"Phikon \citep{phikon}",
         "phikon_v2": r"Phikon-v2 \citep{phikonv2}",
         "plip": r"PLIP \citep{plip}",
         "quiltnet": r"QuiltNet \citep{quiltnet}",
         "retccl": r"RetCCL \citep{retccl}",
         "siglip2": r"SigLIP 2 \citep{siglip2}",
         "uni_v1": r"UNI \citep{uni}",
         "uni_v2": r"UNI v2 \citep{uni}",
         "virchow": r"Virchow \citep{virchow}",
         "virchow2": r"Virchow2 \citep{virchow2}"}
LBL = {e: LABEL[e] for e in E}; MPAR = {e: f"{E[e]['params_M']:.0f}" for e in E}
assert len(LBL) == 57

# tab_hestblocks：去掉 |t| 列，差值改为逐样本差的中位数
L = ["\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
     " & & \\multicolumn{5}{c}{PCC} & \\multicolumn{3}{c}{{\\scshape Octave}} \\\\", "\\cmidrule(lr){3-7}\\cmidrule(lr){8-10}",
     "Encoder & M & Model & Oracle & Ratio & Difference & Cohorts & Model & Oracle & Ratio \\\\", "\\midrule"]
for e in sorted(E, key=lambda e: -E[e]["model_pcc"]):
    o = E[e]
    L.append(f"{LBL[e]} & {MPAR[e]} & {o['model_pcc']:.3f} & {o['oracle_pcc']:.3f} & {pc(o['oracle_ratio'])} & ${o['oracle_minus_model']:+.3f}$ & "
             f"{o['oracle_coh']}/10 & {o['b1_model']:.3f} & {o['b1_oracle']:.3f} & {pc(o['fine_ratio'])} \\\\")
L += ["\\midrule", f"\\textbf{{Median of 57}} & 6--1136 & {med('model_pcc'):.3f} & {med('oracle_pcc'):.3f} & {pc(med('oracle_ratio'))} & ${med('oracle_minus_model'):+.3f}$ & "
      f"{med('oracle_coh'):.0f}/10 & {med('b1_model'):.3f} & {med('b1_oracle'):.3f} & {pc(med('fine_ratio'))} \\\\", "\\bottomrule", "\\end{tabular}"]
wr("tab_hestblocks.tex", L)

# tab_hestsummary
G = S["groups"]
lab = [("All encoders", "all"), ("$<100$M parameters", "lt100"), ("$100$--$600$M", "100to600"), ("$>600$M", "gt600"),
       ("Distilled students ($6$--$87$M)", "distilled"), ("Trained from scratch, $<100$M", "scratch_lt100")]
L = ["\\begin{tabular}{lrrrrrr}", "\\toprule", "Encoders & $n$ & Model PCC & Oracle PCC & Ratio & Range & Cohorts \\\\", "\\midrule"]
for name, g in lab:
    v = G[g]
    L.append(f"{name} & {v['n']} & {v['model_pcc']:.3f} & {v['oracle_pcc']:.3f} & {pc(v['ratio'])} & {100 * v['ratio_range'][0]:.0f}--{100 * v['ratio_range'][1]:.0f}\\% & "
             f"{v['coh_median']:.0f}/10 ({v['n_all10']} at $10/10$) \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
wr("tab_hestsummary.tex", L)

# tab_zscore
L = ["\\begin{tabular}{lrrrrrr}", "\\toprule", "Encoder & Oracle & Oracle, std. & Difference & Ratio & Ratio, std. & Cohorts, raw $\\to$ std. \\\\", "\\midrule"]
for e in sorted(E, key=lambda e: (E[e]["oracle_z_minus_oracle"], e)):
    o = E[e]
    L.append(f"{LBL[e]} & {o['oracle_pcc']:.3f} & {o['oracle_z_pcc']:.3f} & {o['oracle_z_minus_oracle']:+.3f} & {pc(o['oracle_ratio'])} & {pc(o['oracle_z_ratio'])} & {o['oracle_coh']}$\\to${o['oracle_z_coh']} \\\\")
L += ["\\midrule", f"\\textbf{{Median of 57}} & {med('oracle_pcc'):.3f} & {med('oracle_z_pcc'):.3f} & & {pc(med('oracle_ratio'))} & {pc(med('oracle_z_ratio'))} & \\\\",
      "\\bottomrule", "\\end{tabular}"]
wr("tab_zscore.tex", L)

# tab_controls_hest
rng = lambda k: f"{S[k]['range'][0]:.3f}--{S[k]['range'][1]:.3f}"
coh = lambda k: f"{S[k + '_coh']['median']:.0f}/10"
L = ["\\begin{tabular}{lrrr}", "\\toprule", "Predictor on HEST & PCC (median over encoders) & Range over encoders & Cohorts above model \\\\", "\\midrule",
     f"Ridge on frozen features (cohort-selected $\\lambda$) & {S['model_pcc']['median']:.3f} & {rng('model_pcc')} & n/a \\\\",
     f"Image partition, measured means (domain oracle) & {S['oracle_pcc']['median']:.3f} & {rng('oracle_pcc')} & {coh('oracle')} \\\\",
     f"Coordinate partition, measured means & {S['coord_pcc']['median']:.3f} & {rng('coord_pcc')} & {coh('coord')} \\\\",
     f"Random matched-size partition, measured means & {S['random_pcc']['median']:.3f} & {rng('random_pcc')} & {coh('random')} \\\\",
     f"Image partition, means from training samples & {S['train_pcc']['median']:.3f} & {rng('train_pcc')} & {coh('train')} \\\\",
     f"Coordinate partition, means from training samples & {S['train_coord_pcc']['median']:.3f} & {rng('train_coord_pcc')} & {coh('train_coord')} \\\\",
     f"Image partition learned on training samples, means from training samples & {S['inductive_pcc']['median']:.3f} & {rng('inductive_pcc')} & {coh('inductive')} \\\\",
     "\\bottomrule", "\\end{tabular}"]
wr("tab_controls_hest.tex", L)

# tab_methods：PCC、高于该方法的编码器数、与最强编码器的差值（逐样本差的中位数）、高于最强编码器的队列数
M = S["methods"]
names = {"deepspot_hoptimus0_steps500": "DeepSpot, H-optimus-0, 500 steps", "deepspot_hoptimus0": "DeepSpot, H-optimus-0, 10 epochs",
         "deepspot_uni_steps500": "DeepSpot, UNI, 500 steps", "deepspot_uni": "DeepSpot, UNI, 10 epochs", "bleep": "BLEEP \\citep{bleep}",
         "histogene": "HisToGene \\citep{histogene}", "hggep": "HGGEP \\citep{hggep}", "heclip": "HECLIP \\citep{heclip}",
         "hist2st": "Hist2ST \\citep{hist2st}", "thitogene": "THItoGene \\citep{thitogene}"}
L = ["\\begin{tabular}{lrrrr}", "\\toprule", "Method & PCC & Encoders above & Margin & Cohorts above \\\\", "\\midrule"]
for k in sorted(names, key=lambda k: -M[k]["pcc"]):
    v = M[k]
    L.append(f"{names[k]} & {v['pcc']:.3f} & {v['enc_above']}/57 & ${v['margin']:+.3f}$ & {v['coh_above_best']}/10 \\\\")
L += ["\\midrule", f"Frozen encoder and ridge & {rng('model_pcc')} & & & \\\\", "\\bottomrule", "\\end{tabular}"]
wr("tab_methods.tex", L)

# tab_triplex：汇总行由跨队列均值改为跨队列中位数（各队列条目是作者规则下逐 fold 的值，不变）
t = rd("tab_triplex.tex")
rows = [l for l in t.split("\n") if l.count("&") == 6 and not l.startswith(("Cohort", " &", "Mean", "Median"))]
assert len(rows) == 10, len(rows)
val = lambda s: float(s.split("\\rlap")[0])
cols = [[val(r.split("&")[j].strip().rstrip("\\").strip()) for r in rows] for j in range(2, 7)]
medrow = "Median & & " + " & ".join(f"{np.median(c):.3f}" + ("\\rlap{$^{\\dagger}$}" if j == 0 else "") for j, c in enumerate(cols)) + " \\\\"
t2 = "\n".join(medrow if l.startswith(("Mean", "Median")) else l for l in t.split("\n"))
open(f"{PAPER}/tab_triplex.tex", "w").write(t2)
print("写出 6 张表；最强编码器", S["strongest_encoder"], f"{S['strongest_pcc']:.3f}")
print(medrow)
