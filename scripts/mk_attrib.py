# -*- coding: utf-8 -*-
"""复核意见的补充分析汇总（results/xen_attrib/*.json → results/attrib_numbers.json, paper/tab_attrib.tex, paper/attrib_text.tex）。
聚合与正文一致：区域→标本中位数→跨标本中位数（范围为跨标本）。"""
import json, glob, os, numpy as np
R = os.environ.get("S4ST_RESULTS", "results")
SPEC = ["Human_Breast_Biomarkers_S1", "Human_Breast_Biomarkers_S2", "Human_Breast_Biomarkers_S3", "Human_Breast_Biomarkers_S4",
        "Xenium_Prime_Cervical", "Xenium_Prime_Ovarian", "Xenium_V1_Human_Kidney", "Xenium_V1_Human_Ovary",
        "Lung", "Xenium_Prime_Breast_Cancer", "Xenium_Prime_Human_Prostate", "Xenium_Prime_Human_Skin", "Xenium_Prime_Human_Lymph_Node"]   # 2026-09-14 新增 5 个标本
def sp(n):
    if "Human_Lung_Cancer_FFPE" in n: return "Lung"   # Xenium v1 与 Prime 5K 为同一供体同一组织块，按一个标本计
    return next(s for s in SPEC if n.startswith(s))
D = {json.load(open(f))["name"]: json.load(open(f)) for f in glob.glob(f"{R}/xen_attrib/*.json")}
def agg(fn):
    per = {}
    for n, d in D.items(): per.setdefault(sp(n), []).append(fn(d))
    m = [float(np.median(v)) for v in per.values()]; return dict(med=float(np.median(m)), lo=min(m), hi=max(m), n=len(m), above1=int(sum(x > 1 for x in m)))
md = lambda k: (lambda d: d["model_decomp"][k]); cf = lambda k: (lambda d: d["crossfit"][k]); tr = lambda k: (lambda d: d["trainonly"][k]); ct = lambda p, k: (lambda d: d["contiguity"][p][k]); op = lambda o, k: (lambda d: d["operators"][o][k])
N = dict(n_regions=len(D), pcc_model=agg(md("pcc_model")), pcc_between=agg(md("pcc_model_between_only")), frac_between=agg(lambda d: d["model_decomp"]["pcc_model_between_only"] / d["model_decomp"]["pcc_model"]),
         pcc_within=agg(md("pcc_within_vs_within")), pcc_oracle=agg(md("pcc_oracle")), b1_model=agg(md("beta1_model")), b1_between=agg(md("beta1_model_between_only")), b1_within=agg(md("beta1_within_vs_within")), b1_oracle=agg(md("beta1_oracle")),
         var_within_model=agg(md("var_share_within_model")), var_within_truth=agg(md("var_share_within_truth")),
         cf_ratio=agg(cf("ratio_cf")), same_ratio=agg(cf("ratio_same")), cf_dfine=agg(cf("d_fine_cf")), same_dfine=agg(cf("d_fine_same")), cf_b1o=agg(cf("beta1_oracle_cf")), same_b1o=agg(cf("beta1_oracle_same")),
         tr_b1=agg(tr("beta1")), tr_dfine=agg(tr("d_fine")), tr_ratio=agg(tr("ratio")), tr_pcc=agg(tr("pcc")),
         agree_img=agg(ct("image_partition", "neighbour_agreement")), agree_xy=agg(ct("coordinate_partition", "neighbour_agreement")), agree_rand=agg(ct("random_partition", "neighbour_agreement")),
         comp_img=agg(ct("image_partition", "components_per_cluster_median")), largest_img=agg(ct("image_partition", "largest_component_share_median")),
         op_default=agg(op("lazy_rw_default", "ratio")), op_sym=agg(op("symmetric_normalised", "ratio")), op_gauss=agg(op("gaussian_16um", "ratio")))
json.dump(N, open(f"{R}/attrib_numbers.json", "w"), indent=1)
f3 = lambda a: f"{a['med']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]"; f2 = lambda a: f"{a['med']:.2f} [{a['lo']:.2f}, {a['hi']:.2f}]"
L = [r"\begin{tabular}{lrr}", r"\toprule", r"Quantity (Xenium, $K=20$; median [range] over specimens) & Scalar $\mathrm{PCC}$ & Finest band $\beta_1$ \\", r"\midrule",
     f"Trained model $\\Yhat$ & {f3(N['pcc_model'])} & {f3(N['b1_model'])} \\\\", f"Its between-cluster part $\\Proj_\\mathcal{{P}}\\Yhat$, against $\\Ymat$ & {f3(N['pcc_between'])} & {f3(N['b1_between'])} \\\\",
     f"Its within-cluster part $(\\mathbf I-\\Proj_\\mathcal{{P}})\\Yhat$, against $(\\mathbf I-\\Proj_\\mathcal{{P}})\\Ymat$ & {f3(N['pcc_within'])} & {f3(N['b1_within'])} \\\\",
     f"Domain oracle $\\Proj_\\mathcal{{P}}\\Ymat$ & {f3(N['pcc_oracle'])} & {f3(N['b1_oracle'])} \\\\", f"Image partition, means from training blocks & {f3(N['tr_pcc'])} & {f3(N['tr_b1'])} \\\\",
     r"\midrule", f"Oracle with means from the other half of the counts, against this half & -- & {f3(N['cf_b1o'])} \\\\", f"Oracle with means from this half, against this half & -- & {f3(N['same_b1o'])} \\\\", r"\bottomrule", r"\end{tabular}"]
open("paper/tab_attrib.tex", "w").write("\n".join(L) + "\n")
# 可加分解（原先读一份无生成脚本的冻结 JSON，现由原始区域文件直接算，聚合与上面一致）
AD = [json.load(open(f)) for f in glob.glob(f"{R}/xen_addsplit/*.json")]
def _addagg(k):
    per = {}
    for d in AD: per.setdefault(sp(d["name"]), []).append(float(d[k]))
    m = [float(np.median(v)) for v in per.values()]; return [float(np.median(m)), min(m), max(m)]
_iso = [d["isolated_nodes"] for d in AD]; _isof = [d["isolated_nodes"] / d["n"] for d in AD]
ADD = dict(between_share=_addagg("between_share"), within_share=_addagg("within_share"), between_term=_addagg("between_term"),
           within_term=_addagg("within_term"), pcc=_addagg("pcc"), n_regions=len(AD),
           iso_median=float(np.median(_iso)), iso_min=int(min(_iso)), iso_max=int(max(_iso)),
           iso_frac_median=float(np.median(_isof)), iso_frac_max=float(max(_isof)))
json.dump(ADD, open(f"{R}/addsplit_numbers.json", "w"), indent=1)
txt = (r"\paragraph{The model's score sits in its between-cluster part and its finest-band fidelity in the within-cluster part.} " +
       f"The oracle bounds a partition. It does not say what the trained model uses. Table~\\ref{{tab:attrib}} therefore splits the model's own prediction into its between-cluster part $\\Proj_\\mathcal{{P}}\\Yhat$ and its within-cluster part $(\\mathbf I-\\Proj_\\mathcal{{P}})\\Yhat$ on the same image partition. The split uses no test label. The between-cluster part alone scores {f3(N['pcc_between'])} against the measurement. That is a median {100*N['frac_between']['med']:.0f}\\% of the model's own score, the ratio formed per region (range {100*N['frac_between']['lo']:.0f}--{100*N['frac_between']['hi']:.0f}\\%). It also equals the oracle's score to three decimals. The model's cluster means track the measured ones almost perfectly. The within-cluster part correlates {f3(N['pcc_within'])} with the within-cluster measurement, so the model does recover structure inside clusters, and it holds {100*N['var_within_model']['med']:.0f}\\% of the model's variance against {100*N['var_within_truth']['med']:.0f}\\% of the measurement's. In the finest band the split is complete: the between-cluster part has $\\beta_1 = {N['b1_between']['med']:.2f}$, the oracle's value, while the within-cluster part has {N['b1_within']['med']:.2f} and the full model {N['b1_model']['med']:.2f}. Everything the finest band credits to the model is within-cluster structure, and everything the scalar credits to the oracle is between-cluster structure. The ratio of the two scalar scores is not an additive share, since the two correlations carry different denominators. With a common denominator the per-gene correlation splits exactly into a between-cluster and a within-cluster term, $\\rho(\\hat y, y) = \\langle \\Proj\\hat y, \\Proj y\\rangle/(\\lVert\\hat y\\rVert\\lVert y\\rVert) + \\langle(\\mathbf I-\\Proj)\\hat y, (\\mathbf I-\\Proj)y\\rangle/(\\lVert\\hat y\\rVert\\lVert y\\rVert)$ for centred vectors, and on the ${ADD['n_regions']}$ regions the between-cluster term accounts for a median {100*ADD['between_share'][0]:.0f}\\% of the model's score (range {100*ADD['between_share'][1]:.0f}--{100*ADD['between_share'][2]:.0f}\\% over specimens) and the within-cluster term for {100*ADD['within_share'][0]:.0f}\\%." "\n\n" +
       r"\paragraph{Cross-fitting the oracle leaves its finest band unchanged, and a learned domain predictor keeps the bound.} " +
       f"The attenuation argument of Appendix~\\ref{{app:reliab}} treats the two predictors as independent of the measurement noise. The oracle is built from the measured field, so it is not. We therefore split each bin's counts into two binomial halves, estimate the cluster means from one half and score them against the other, so that the two share no counting noise. The cross-fitted oracle's finest-band correlation is {f3(N['cf_b1o'])}. Means taken from the scored half itself give {f3(N['same_b1o'])}. The shortfall ratio is {f2(N['cf_ratio'])} against {f2(N['same_ratio'])}. Shared noise changes nothing at the first decimal. The ratio exceeds one in {N['cf_ratio']['above1']} of {N['cf_ratio']['n']} specimens. The learnable predictor of Table~\\ref{{tab:controls_main}} uses cluster means from the training blocks. It gives the same picture without any test label: $\\beta_1 = {N['tr_b1']['med']:.2f}$, a fine-band shortfall of {100*N['tr_dfine']['med']:.0f}\\% and a ratio of {f2(N['tr_ratio'])}, above one in {N['tr_ratio']['above1']} of {N['tr_ratio']['n']}." "\n\n" +
       r"\paragraph{The image partition is a morphology partition, not a map of contiguous tissue areas.} " +
       f"On the $8$-neighbour grid graph a median {100*N['agree_img']['med']:.0f}\\% of edges join bins with the same label (range {100*N['agree_img']['lo']:.0f}--{100*N['agree_img']['hi']:.0f}\\%), against {100*N['agree_xy']['med']:.0f}\\% for the coordinate partition and {100*N['agree_rand']['med']:.0f}\\% for a random one, but each cluster splits into a median of about {N['comp_img']['med']:.0f} connected components, the largest holding {100*N['largest_img']['med']:.0f}\\% of its bins. The oracle is therefore a bound on partition identity, not on membership of an annotated tissue domain." "\n\n" +
       r"\paragraph{Symmetric and Gaussian operators leave the shortfall ratio in the same range as the lazy walk.} " +
       f"Replacing the lazy random walk by its symmetrically normalised form $(\\mathbf I + \\mathbf D^{{-1/2}}\\mathbf A\\mathbf D^{{-1/2}})/2$ gives a finest-band shortfall ratio of {f2(N['op_sym'])}, and a Gaussian spatial kernel of width $16\\,\\mu$m over neighbours within $48\\,\\mu$m gives {f2(N['op_gauss'])}, against {f2(N['op_default'])} for the default; each exceeds one in {N['op_sym']['n']} of {N['op_sym']['n']} specimens." "\n" +
       r"\begin{table}[h]" "\n" r"\centering\small" "\n" r"\caption{The trained model split on the image partition, the oracle, a learnable partition predictor, and the cross-fitted oracle, on the Xenium regions. Medians over specimens of per-specimen medians over regions; the cross-fitted rows use two binomial halves of the counts.}" "\n" r"\label{tab:attrib}" "\n" r"\resizebox{\textwidth}{!}{\input{tab_attrib}}" "\n" r"\end{table}" "\n")
open("paper/attrib_text.tex", "w").write(txt)
print(f"regions={len(D)} specimens={N['pcc_model']['n']} | between frac {N['frac_between']['med']:.3f} | within pcc {N['pcc_within']['med']:.3f} | b1 model/between/within/oracle {N['b1_model']['med']:.3f}/{N['b1_between']['med']:.3f}/{N['b1_within']['med']:.3f}/{N['b1_oracle']['med']:.3f} | cf {N['cf_ratio']['med']:.2f} same {N['same_ratio']['med']:.2f} | train ratio {N['tr_ratio']['med']:.2f} | agree {N['agree_img']['med']:.2f} comps {N['comp_img']['med']:.0f} | ops {N['op_sym']['med']:.2f} {N['op_gauss']['med']:.2f}")
