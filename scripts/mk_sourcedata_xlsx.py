# -*- coding: utf-8 -*-
"""把每张图背后的逐单元数值汇成一个 source-data 工作簿。所有数取自冻结的 JSON。"""
import json, glob, os, re
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "paper", "source_data.xlsx")
J = lambda p: json.load(open(os.path.join(R, p)))
def spec(n):
    m = re.match(r"(Human_Breast_Biomarkers_S\d)_(Top|Mid|Bot)$", n)
    if "Human_Lung_Cancer_FFPE" in n: return "Lung"   # v1 与 Prime 5K 同一供体同一组织块，按一个标本计
    return m.group(1) if m else n

wb = Workbook(); wb.remove(wb.active)
IDX = []
HEAD = PatternFill("solid", fgColor="E8E8E8")

def sheet(name, cols, rows, panel, src, note=""):
    ws = wb.create_sheet(name[:31])
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(bold=True); c.fill = HEAD; c.alignment = Alignment(wrap_text=True, vertical="top")
    for r in rows: ws.append(list(r))
    ws.freeze_panes = "A2"
    for i, c in enumerate(cols, 1):
        w = len(str(c))
        for r in rows[:200]:
            if i <= len(r): w = max(w, len(str(r[i - 1])[:18]))
        ws.column_dimensions[get_column_letter(i)].width = min(max(w + 2, 9), 30)
    IDX.append((name[:31], panel, len(rows), src, note))
    print(f"  {name[:31]:24s} {len(rows):5d} x {len(cols):3d}  {panel}")

# ═══ 图 1
d = J("blocks_vs_detail.json"); rows = []
for lay, key in (("per-gene mean (50 genes)", "per_gene"), ("PC1 projection of the map", "map_pc1")):
    D = d[key]
    for who in ("truth", "ridge", "dom20"):
        rows.append([lay, {"truth": "measurement", "ridge": "trained model", "dom20": "block oracle"}[who],
                     D.get("pcc", {}).get(who), D.get("sigma_um", {}).get(who),
                     D.get("band_pcc", {}).get(who), D.get("band_var_share", {}).get(who)])
sheet("Fig1_window", ["quantity", "field", "PCC with measurement", "equivalent sigma (um)",
                      "finest-band PCC", "finest-band variance share"], rows,
      "Fig 1b-e, g", "results/blocks_vs_detail.json",
      f"one {d['roi']['side_um']:.0f} um window, {d['roi']['n_bins']} bins, {d['roi']['n_domains']} domains; "
      f"median-gradient window of {d['roi']['percentiles']['n_windows']}; PC1 holds {100*d['map_pc1']['pc1_var_share']:.1f}% of measured variance")

# ═══ 图 2a：逐样本的落差-带宽剖面
F = J("fig2_scale.json"); sg = F["sigma_um"]
rows = [["median across specimens", ""] + [round(v, 4) for v in F["gap_median"]]]
for k, v in sorted(F["by_specimen"].items()):
    rows.append([k, round(v["scalar"], 4)] + [round(x, 4) for x in v["gap"]])
sheet("Fig2a_band_profile", ["specimen", "scalar shortfall (%)"] + [f"sigma={s:.1f} um" for s in sg], rows,
      "Fig 2a", "results/fig2_scale.json", "block-oracle shortfall relative to the trained model, per cent, by band width")

# ═══ 图 2b/2c：逐区域
rows2b, rows2c = [], []
for f in sorted(glob.glob(os.path.join(R, "blocks_xen_bands", "*.json"))):
    d = json.load(open(f)); P = d["pred"]; n = d["name"]
    ts = sorted(int(t) for t in P["ridge"]["bands"])
    ov = 100 * (P["ridge"]["pcc"] - P["dom20"]["pcc"]) / P["ridge"]["pcc"]
    r0, b0 = P["ridge"]["bands"][str(ts[0])], P["dom20"]["bands"][str(ts[0])]
    fi = 100 * (r0 - b0) / r0
    rows2b.append([n, spec(n), P["ridge"]["pcc"], P["dom20"]["pcc"], P["ridge"]["sigma_um"],
                   P["dom20"]["sigma_um"], ov, fi, fi / ov])
    for t in ts:
        rows2c.append([n, spec(n), d["sigma_um"][str(t)], t] +
                      [P[w]["band_var"][str(t)] for w in ("truth", "ridge", "dom20")] +
                      [P[w]["bands"][str(t)] for w in ("ridge", "dom20")])
sheet("Fig2b_specimens", ["region", "specimen", "PCC trained model", "PCC block oracle",
      "sigma model (um)", "sigma oracle (um)", "scalar shortfall (%)", "finest-band shortfall (%)", "ratio"],
      rows2b, "Fig 2b, Table 1", "results/blocks_xen_bands/*.json",
      "ratios are formed per region, then aggregated within a specimen")
sheet("Fig2c_variance", ["region", "specimen", "sigma (um)", "ladder step t", "variance share measured",
      "variance share model", "variance share oracle", "band PCC model", "band PCC oracle"],
      rows2c, "Fig 2c", "results/blocks_xen_bands/*.json", "each band renormalises; band PCC is not a decomposition of the scalar PCC")

# ═══ 图 3 / 表 2、3、5：编码器 x k
# 2026-09-01：检索地板线已撤出论文（见 CLAIMS.md）。以下三块只在设了
# S4ST_FLOOR_SHEETS=1 时生成，证据本身冻结在 results/ 的 JSON 里。
if os.environ.get("S4ST_FLOOR_SHEETS") == "1":
  K = J("k_sensitivity.json"); rows = []
  for k in sorted(K, key=int):
      for e, r in sorted(K[k].items()):
          rows.append([e, int(k), r["floor"] + r["gap"], r["floor"], r["gap"],
                       r["rel"], r["rel_ok"], r["n_cohorts_win"], r["n_cohorts_ok"], r["n_cohorts_win_ok"], r["P"], r["P_ok"]])
  sheet("Fig3_benchmark", ["encoder", "k", "reported PCC (ridge)", "matched k-NN floor", "gap",
        "margin (%)", "margin, usable cohorts only (%)", "cohorts won / 10", "cohorts usable",
        "cohorts won among usable", "P", "P, usable cohorts only"], rows,
        "Fig 3, Tables 2/3/5", "results/k_sensitivity.json",
        "reported PCC is constant across k by construction; only the floor moves")

  # ═══ 附录表 8：替代回归头扫描（三个编码器 × 十个队列 fold 0 × 各头的训练/测试 PCC）
  import glob as _gl
  sw = {}
  for f in sorted(_gl.glob(os.path.join(R, "mlp_head_sweep*.json"))):
      for e, d in json.load(open(f)).items():
          for c, v in d.items():
              sw.setdefault(e, {}).setdefault(c, {"n": v["n_train_sections"], "v": {}})["v"].update(v["variants"])
  if sw:
      rows = []
      for e in sw:
          for c in sorted(sw[e]):
              rg = sw[e][c]["v"].get("ridge", {}).get("test_pcc")
              for h, v in sorted(sw[e][c]["v"].items()):
                  rows.append([e, c, sw[e][c]["n"], h, v["train_pcc"], v["test_pcc"],
                               (v["test_pcc"] - rg) if rg is not None else None])
      sheet("Tab8_heads_sweep", ["encoder", "cohort", "training sections", "head", "train PCC", "test PCC", "test PCC minus ridge"],
            rows, "Table 8 (appendix)", "results/mlp_head_sweep*.json",
            "fold 0 of each cohort; heads share the StandardScaler->PCA-256 input; mlp_groupval holds out one training section for early stopping and is excluded from the paper table")

  # ═══ 附录表 9：最佳非线性头在全部 30 个编码器上（标签 MLP_TAG，默认 mlp10）
  for MT in os.environ.get("MLP_TAGS", "mlp10,mlp100,rsel").split(","):
    kp, cp = os.path.join(R, f"k_sensitivity_{MT}.json"), os.path.join(R, f"cohort_spread_{MT}.json")
    if os.path.exists(kp) and os.path.exists(cp):
        KM, CM = json.load(open(kp)), json.load(open(cp)); rows = []
        for k in sorted(KM, key=int):
            for e, r in sorted(KM[k].items()):
                seeds = [json.load(open(f))["mean_pcc"] for f in sorted(_gl.glob(os.path.join(R, "mlp_seeds", f"hest_{MT}_ps_{e}_s*.json")))]
                rows.append([e, int(k), r["floor"] + r["gap"], r["floor"], r["gap"], r["rel"], r["rel_ok"], r["n_cohorts_win"], r["P"], r["P_ok"],
                             CM[e]["cohort_mean"] if k == "50" else None, CM[e]["cohort_sem"] if k == "50" else None, CM[e]["t"] if k == "50" else None,
                             len(seeds), (float(np.std(seeds, ddof=1)) if len(seeds) > 1 else None)] + seeds)
        sheet(f"Tab9_{MT}_all30", ["encoder", "k", "MLP PCC (seed 0)", "matched k-NN floor", "gap", "margin (%)", "margin, usable cohorts only (%)",
              "cohorts won / 10", "P", "P, usable cohorts only", "cohort-level mean (k=50)", "cohort-level sem (k=50)", "t (k=50)",
              "n seeds", "seed sd of mean PCC", "seed 0", "seed 1", "seed 2"], rows,
              "Table 9 (appendix)", f"results/k_sensitivity_{MT}.json, results/cohort_spread_{MT}.json, results/mlp_seeds/",
              "same folds, genes, target and floors as Fig3_benchmark; only the head differs")

# ═══ 图 4：换算率
CN = {"深度线 · 25 个图像编码器（跨片）": "depth line, 25 image encoders, cross-section",
      "方法 · 片内块 CV（旧 200 基因基准）": "methods, within-section block CV (earlier 200-gene protocol)",
      "方法 · 跨片留一（旧 200 基因基准）": "methods, leave-one-section-out (earlier 200-gene protocol)",
      "多尺度分箱（对照）": "multi-scale binning (control)"}
rows = [[CN.get(c["name"], c["name"]), c.get("n"), c.get("slope_lnsigma_per_pcc"), c.get("pct_per_0.01pcc"),
         c.get("r2"), c.get("pcc_lo"), c.get("pcc_hi"), c.get("sigma_lo"), c.get("sigma_hi")]
        for c in J("conversion_rate.json")]
sheet("Fig4_conversion", ["series", "n", "d ln sigma / d PCC", "% sigma per +0.01 PCC", "R2",
      "PCC min", "PCC max", "sigma max (um)", "sigma min (um)"], rows,
      "Fig 4g-h", "results/conversion_rate.json", "the control row is the only one whose sign is positive: coarsening the output grid alone")

# ═══ 标定阶梯
C = J("ceiling_hvg50_t2048.json"); rows = [[int(t), C["sigma_um"][t], C["c_half"][t], C["c_full"][t]] for t in sorted(C["sigma_um"], key=int)]
sheet("Calibration_ladder", ["ladder step t", "sigma (um)", "ceiling, half sample", "ceiling, full sample"],
      rows, "Fig 1f, Eq. 3", "results/ceiling_hvg50_t2048.json", "sigma is proportional to t^0.49 over three orders of magnitude")

# ═══ 图 4 的 c–f：固定栅格下两个判据的最优点，以及划分几何的敏感度
ssp = os.path.join(R, "split_sensitivity.json")
if os.path.exists(ssp):
    SS = json.load(open(ssp))
    c = SS["panel_c"]
    sheet("Fig4c_optimum", ["prediction bin (um)", "median PCC, grid fixed at 16 um",
          "median sigma, grid follows (um)", "is best PCC", "is best sigma"],
          [[b, c["pcc_fixed_grid"].get(str(b)), c["sigma_follow_grid"].get(str(b)),
            b == c["best_bin_pcc"], b == c["best_bin_sigma"]] for b in c["bins_um"]],
          "Fig 4c", "results/split_sensitivity.json",
          "the two criteria place their optimum at different prediction bins")
    d = SS["panel_def"]
    sheet("Fig4def_split_grid", ["spatial block grid", "median PCC", "median sigma (um)"],
          [[f"{g}x{g}", d["pcc"][str(g)], d["sigma"][str(g)]] for g in d["grids"]],
          "Fig 4d-f", "results/split_sensitivity.json",
          f"16x16 to 2x2 moves PCC by {d['pcc_change_pct']:.2f}% and sigma by {d['sigma_change_pct']:.2f}%, "
          f"a sensitivity ratio of {d['sensitivity_ratio']:.1f}")

# ═══ 附录图 7 的六个下游读数：逐读数的退化统计
dsp = os.path.join(R, "downstream_summary.json")
if os.path.exists(dsp):
    DSS = json.load(open(dsp))
    sheet("FigS3_readout_summary", ["readout", "panel", "source", "median at 16 um", "median at 64 um",
          "regions", "regions degraded", "specimens", "specimens degraded", "P", "worse direction"],
          [[k, v["panel"], v["source"], v["v16"], v["v64"], v["n_regions"], v["regions_worse"],
            v["n_specimens"], v["specimens_worse"], v["P"], v["worse_direction"]]
           for k, v in DSS.items()],
          "Fig 7", "results/downstream_summary.json",
          "worse direction -1 means a smaller value is worse; inference is at the specimen level")

# ═══ 表 6 的消融：三条轴
apth = os.path.join(R, "ablate_spread.json")
if os.path.exists(apth):
    AB = json.load(open(apth))
    rows = []
    for ax in ("alpha", "ngene", "K"):
        if ax not in AB: continue
        for k in sorted(AB[ax]["levels"], key=float):
            o = AB[ax]["levels"][k]
            rows.append([ax, k, (k == AB[ax]["default"]), o["n_specimens"], o["share"], o["scalar"],
                         o["fine"], o["ratio"], o["ratio_min"], o["ratio_max"], o["seed_sd_median"],
                         f"{o['specimens_agree']}/{o['n_specimens']}", o["P"]])
    sheet("Table6_ablation", ["axis", "level", "is default", "specimens", "block-oracle share (%)",
          "scalar shortfall (%)", "finest-band shortfall (%)", "ratio", "ratio min across specimens",
          "ratio max across specimens", "median seed s.d.", "specimens agreeing", "P"], rows,
          "Table 6", "results/ablate_spread.json",
          "one axis varied at a time, the others held at their defaults; each cell is a median over 3 K-means seeds")
    rows = []
    for f in sorted(glob.glob(os.path.join(R, "blocks_xen_ablate", "*.json"))):
        d = json.load(open(f))
        for ax in ("alpha", "ngene", "K"):
            for k, rs in d.get(ax, {}).items():
                for r in rs:
                    rows.append([d["name"], spec(d["name"]), ax, k, r["seed"], r["ridge_pcc"],
                                 r["ridge_band"], r["oracle_pcc"], r["oracle_band"], r["overall"],
                                 r["fineband"], r["ratio"], r["share"]])
    sheet("Table6_ablation_detail", ["region", "specimen", "axis", "level", "seed", "ridge PCC",
          "ridge finest-band PCC", "oracle PCC", "oracle finest-band PCC", "scalar shortfall (%)",
          "finest-band shortfall (%)", "ratio", "block-oracle share (%)"], rows,
          "Table 6", "results/blocks_xen_ablate/*.json", "one row per region, axis level and seed")

# ═══ Visium HD 四折的块预言机占比（摘要与 §5.1 引用）
dp = os.path.join(R, "dom_sigma.json")
if os.path.exists(dp):
    DS = json.load(open(dp))
    rows = [[f, DS["R_ridgeHEST"][f]["pcc"], DS["D_domImg_k20"][f]["pcc"],
             100 * DS["D_domImg_k20"][f]["pcc"] / DS["R_ridgeHEST"][f]["pcc"],
             DS["D_domImg_k200"][f]["pcc"],
             100 * DS["D_domImg_k200"][f]["pcc"] / DS["R_ridgeHEST"][f]["pcc"],
             DS["R_ridgeHEST"][f]["sigma_um"], DS["D_domImg_k20"][f]["sigma_um"]]
            for f in sorted(DS["D_domImg_k20"])]
    sheet("VisiumHD_block_share", ["fold", "ridge PCC", "block oracle PCC (K=20)", "share K=20 (%)",
          "block oracle PCC (K=200)", "share K=200 (%)", "sigma ridge (um)", "sigma oracle (um)"],
          rows, "abstract, Sec 5.1", "results/dom_sigma.json",
          "four within-slide folds of two Visium HD colon sections")

# ═══ 表 1 的带宽依赖：比值随「最细带」的选择怎么变
import re as _re
from math import comb as _comb
def _sp(n):
    m = _re.match(r"(Human_Breast_Biomarkers_S\d)_(Top|Mid|Bot)$", n)
    return m.group(1) if m else n
def _p(k, n):
    k = min(k, n - k)
    return min(1.0, 2 * sum(_comb(n, i) for i in range(k + 1)) / 2 ** n)
BB = {}
for f in sorted(glob.glob(os.path.join(R, "blocks_xen_bands", "*.json"))):
    d = json.load(open(f)); BB[d["name"]] = (d["pred"], d["sigma_um"])
ts = sorted(int(x) for x in list(BB.values())[0][1])
rows = []
for t_ in ts:
    per = {}
    for n, (P, sg) in BB.items():
        ov = 100 * (P["ridge"]["pcc"] - P["dom20"]["pcc"]) / P["ridge"]["pcc"]
        r0, b0 = P["ridge"]["bands"][str(t_)], P["dom20"]["bands"][str(t_)]
        per.setdefault(_sp(n), []).append((ov, 100 * (r0 - b0) / r0))
    o = [np.median([x[0] for x in v]) for v in per.values()]
    fi = [np.median([x[1] for x in v]) for v in per.values()]
    rt = [np.median([x[1] / x[0] for x in v]) for v in per.values()]
    w = sum(1 for a_, b_ in zip(o, fi) if b_ > a_)
    rows.append([t_, list(BB.values())[0][1][str(t_)], float(np.median(o)), float(np.median(fi)),
                 float(np.median(rt)), float(min(rt)), float(max(rt)), f"{w}/{len(o)}", _p(w, len(o))])
sheet("Table1_band_choice", ["ladder step t", "sigma (um)", "scalar shortfall (%)",
      "band shortfall (%)", "ratio median", "ratio min", "ratio max", "specimens agreeing", "P"],
      rows, "Table 1 sensitivity", "results/blocks_xen_bands/*.json",
      "which band is called the finest; the data are binned to 16 um, so the first rung sits below the bin pitch")

# ═══ β₁ 是否比 PCC 更贴合下游效用（两次检验，均不支持；见 CLAIMS B10）
bv = os.path.join(R, "blur_verdict.json"); ba = os.path.join(R, "blur_ds_all.json")
if os.path.exists(ba):
    BA = json.load(open(ba))
    ts = sorted((int(x) for x in list(BA.values())[0]), key=int)
    cols = ["region", "specimen", "blur t", "PCC", "beta1", "svg_top_jaccard", "svg_rank_rho",
            "hotspot_jaccard", "boundary_shift_um", "coloc_preserve", "hotspot_recall_selectivity"]
    rows = []
    for n, L in sorted(BA.items()):
        for t_ in ts:
            v = L[str(t_)]
            rows.append([n, spec(n), t_] + [v.get(k) for k in cols[3:]])
    sheet("StepFour_blur", cols, rows, "not in the paper as a claim",
          "results/blur_ds_*.json",
          "the prediction is blurred on a fixed grid; training data, target and grid are unchanged")
if os.path.exists(bv):
    BV = json.load(open(bv))
    sheet("StepFour_verdict", ["readout", "specimens", "retained at last blur", "distance to PCC",
          "distance to beta1", "specimens where beta1 is closer", "P"],
          [[k, v["n"], v["keep_last"], v["dist_pcc"], v["dist_beta"], v["n_beta_closer"], v["P"]]
           for k, v in BV.items()],
          "not in the paper as a claim", "results/blur_verdict.json",
          "0 of 5 readouts track beta1 more closely than PCC; the paper makes no downstream-utility claim")

# ═══ 表 7：五个已发表方法，逐样本原值与汇总
# ═══ 表 1（正文）块预言机 × 49 个编码器：K=20/50/200，官方 α 与留一队列选 α 两种分母（正文用选 α）
bsp = os.path.join(R, "hest_blocks_summary.json"); epp = os.path.join(R, "encoder_params.json")
if os.path.exists(bsp) and os.path.exists(epp):
    BS = json.load(open(bsp)); EP = json.load(open(epp))
    cols = ["encoder", "params", "n_samples", "n_cohorts"]
    keys = ["blk_k20_sel", "blk_k20_off", "blk_k50_sel", "blk_k50_off", "blk_k200_sel", "blk_k200_off"]
    for k in keys:
        cols += [f"{k}:model PCC", f"{k}:oracle PCC", f"{k}:ratio %", f"{k}:cohort mean diff", f"{k}:cohort sem", f"{k}:t", f"{k}:cohorts won", f"{k}:P"]
    rows = []
    for d in sorted(BS, key=lambda d: -d["blk_k20_sel"]["mod"]):
        r = [d["enc"], EP.get(d["enc"], {}).get("params"), d["n_samples"], d["n_cohorts"]]
        for k in keys:
            v = d[k]; r += [v["mod"], v["blk"], v["share"], v["cohort_mean"], v["cohort_sem"], v["t"], v["won"], v["P"]]
        rows.append(r)
    sheet("Table1_domain_oracle", cols, rows, "Table 1", "results/hest_blocks_summary.json; results/encoder_params.json",
          "Domain oracle (measured cluster means on the test section; K-means on PCA-50 of the encoder's own features) vs ridge on frozen features. "
          "_sel: ridge alpha chosen leave-one-cohort-out (used in Table 1); _off: benchmark's official alpha. "
          "ratio % = oracle/model x 100; cohort mean diff = mean over cohorts of the per-cohort median paired difference (PCC units); "
          "P = exact two-sided sign test over 10 cohorts. params = all parameters of the loaded module (CLIP-family counts include the text tower).")
# ═══ 表 1 稳健性：PCA 前各维 z-score 的域预言机 vs 标准配方（同一分母）
zsp = os.path.join(R, "hest_blocks_z_summary.json")
if os.path.exists(zsp):
    ZS = json.load(open(zsp)); cols = ["encoder"]
    for k in ("k20", "k50", "k200"):
        cols += [f"{k}:model", f"{k}:oracle", f"{k}:oracle zscored", f"{k}:ratio %", f"{k}:ratio % zscored",
                 f"{k}:cohort mean diff", f"{k}:cohort mean diff zscored", f"{k}:cohorts won", f"{k}:cohorts won zscored"]
    rows = []
    for r in sorted(ZS, key=lambda r: r["enc"]):
        row = [r["enc"]]
        for k in ("k20", "k50", "k200"):
            v = r[k]; row += [v["model"], v["orc"], v["orc_z"], v["share"], v["share_z"], v["std"][0], v["zs"][0], v["std"][2], v["zs"][2]]
        rows.append(row)
    sheet("Table1_zscore_robustness", cols, rows, "Table 1 robustness", "results/hest_blocks_z_summary.json",
          "Same domain oracle with each feature dimension standardised (zero mean, unit variance) before PCA(50); the standard recipe uses raw features. "
          "Denominator is the leave-one-cohort-out ridge in both columns. OpenMidnight's features carry one dimension with ~91% of the total variance, "
          "which dominates the unstandardised partition at K=20; standardisation moves no other encoder's ratio by more than 3 points.")
mvp = os.path.join(R, "methods_vs_floor.json")
if os.path.exists(mvp):
    MV = json.load(open(mvp))
    ref = MV.pop("_reference")
    sheet("Table7_methods", ["method", "samples", "PCC (sample mean)", "below encoders",
          "encoders", "below floors", "floors", "vs best encoder", "cohort mean", "cohort s.e.m.",
          "cohorts won", "cohorts", "P"],
          [[k, v["n_samples"], v["mean_sample"], v["n_below_encoders"], v["n_encoders"],
            v["n_below_floors"], v["n_floors"], v["vs_best_encoder"], v["cohort_mean"],
            v["cohort_sem"], v["cohorts_won"], v["n_cohorts"], v["P"]] for k, v in MV.items()],
          "Table 7", "results/methods_vs_floor.json",
          "each method run under the benchmark's own protocol: same splits, same 50 genes, same target")
    FILES = {"HisToGene": "histogene_matched.json", "Hist2ST": "hist2st_matched_all.json",
             "BLEEP": "bleep_hest.json", "HECLIP": "heclip_hest.json", "HGGEP": "hggep_hest.json",
             "THItoGene": "thitogene_hest.json"}
    rows = []
    for m, f in FILES.items():
        fp = os.path.join(R, f)
        if not os.path.exists(fp): continue
        for sid, d in sorted(json.load(open(fp)).items()):
            rows.append([m, sid, d.get("cohort"), d.get("pcc"), len(d.get("folds", []))])
    sheet("Table7_methods_sample", ["method", "sample", "cohort", "PCC", "folds"], rows,
          "Table 7", "results/{histogene,hist2st,bleep,heclip,hggep}_*.json",
          "one row per method and sample; 72 samples each")
    MV["_reference"] = ref

# ═══ 表 2 的误差棒：逐队列
CS = json.load(open(os.path.join(R, "cohort_spread.json")))
rows = []
for e, o in sorted(CS.items(), key=lambda kv: -kv[1]["model_mean_sample"]):
    for c in o["by_cohort"]:
        rows.append([e, c["cohort"], c["n_samples"], c["mean_model"], c["mean_floor"],
                     c["median_paired_diff"], o["cohort_mean"], o["cohort_sem"], o["t"], o["cohorts_won"]])
sheet("Table2_cohort_spread", ["encoder", "cohort", "samples", "mean model PCC", "mean floor PCC",
      "median paired difference", "encoder cohort mean", "encoder cohort s.e.m.", "encoder t", "cohorts won"],
      rows, "Table 2 error bar", "results/cohort_spread.json",
      "the per-cohort statistic is the median paired per-sample difference, the same one the sign test counts")

# ═══ 表 1 的误差棒：KMeans 种子
sp = os.path.join(R, "seed_spread.json")
if os.path.exists(sp):
    SS = json.load(open(sp))
    sheet("Table1_seed_headline", ["seed", "headline ratio (8-specimen median)", "min across specimens",
          "max across specimens", "scalar shortfall (%)", "finest-band shortfall (%)",
          "block-oracle share (%)", "specimens agreeing", "P"],
          [[h["seed"], h["ratio_median"], h["ratio_min"], h["ratio_max"], h["scalar_median"],
            h["fine_median"], h["share_median"], f"{h['specimens_agree']}/{h['n_specimens']}", h["P"]]
           for h in SS["by_seed"]],
          "Table 1 error bar", "results/seed_spread.json",
          "each seed reruns the whole aggregation: ratio per region, median within specimen, median across specimens")
    rows = []
    for f in sorted(glob.glob(os.path.join(R, "blocks_xen_seeds", "*.json"))):
        d = json.load(open(f))
        for r in d["seeds"]:
            rows.append([d["name"], spec(d["name"]), r["seed"], r["dom_pcc"], r["dom_band_pcc"],
                         r["overall"], r["fineband"], r["ratio"], r["share"],
                         r["ari_vs_seed0"], r["inertia"], r["n_clusters_used"]])
    sheet("Table1_seed_detail", ["region", "specimen", "seed", "oracle PCC", "oracle finest-band PCC",
          "scalar shortfall (%)", "finest-band shortfall (%)", "ratio", "block-oracle share (%)",
          "ARI vs seed 0", "K-means inertia", "clusters used"], rows,
          "Table 1 error bar", "results/blocks_xen_seeds/*.json",
          "one row per region and seed; the ridge scores it is compared against are deterministic and sit in Fig2b_specimens")
    sheet("Table1_seed_region", ["region", "seeds", "ratio median", "ratio min", "ratio max", "ratio s.d.",
          "scalar shortfall median (%)", "finest-band shortfall median (%)", "block-oracle share median (%)",
          "ARI vs seed 0, median", "seeds with fine > scalar"],
          [[n, v["n_seeds"], v["ratio_median"], v["ratio_min"], v["ratio_max"], v["ratio_sd"],
            v["overall_median"], v["fineband_median"], v["share_median"], v["ari_median"],
            f"{v['fineband_gt_overall']}/{v['n_seeds']}"] for n, v in sorted(SS["by_region"].items())],
          "Table 1 error bar", "results/blocks_xen_seeds/*.json",
          "ARI below 1 means the partition itself changed between seeds")
else:
    print("  (seed_spread.json 尚未生成，跳过两张种子表)")

# ═══ 由集群抽取的表（含附录图）
SD = json.load(open(os.path.join(R, "sourcedata.json")))
PANEL = {"Fig4_grid": "Fig 4a-b, Table 4", "FigS1_moran_region": "Fig 5", "FigS1_moran_gene": "Fig 5",
         "FigS2_knobs": "Fig 6", "FigS2_target": "Fig 6", "FigS3_downstream": "Fig 7"}
def flat(x, inner=None):
    """tower/panel/ctx 扫描的 _summary 按方法再嵌一层；取 Ridge_HEST 这一支。"""
    if isinstance(x, dict):
        y = x.get("Ridge_HEST", x.get(next(iter(x)), None))
        return flat(y[inner] if inner and isinstance(y, dict) and inner in y else y)
    return None if isinstance(x, float) and x != x else x

for k, v in SD.items():
    cols, rows = v["columns"], v["rows"]
    if k == "FigS2_knobs":
        cols = ["knob", "setting", "PCC (ridge)", "PCC (image k-NN)", "sigma, ridge (um)", "dim", "Moran ratio (ridge)"]
        rows = [[r[0], r[1], flat(r[2]),
                 (r[2].get("imageKNN") if isinstance(r[2], dict) else None),
                 flat(r[3], "pcc"), r[4], flat(r[5])] for r in rows]
    else:
        rows = [[flat(x) if isinstance(x, (dict, float)) else x for x in r] for r in rows]
    sheet(k, cols, rows, PANEL.get(k, ""), v["source"], v.get("note", ""))

# ═══ 索引页放最前

# ── 2026-09-05 审稿意见驱动的新结果 ──
def _load_dir(d): return {json.load(open(f))["name"]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, d, "*.json")))}
_rows = []
for _tag in ["base", "k4", "k12", "lazy75"]:
    for _n, _x in _load_dir(f"blocks_xen_bands_{_tag}").items():
        _op = _x.get("operator", {}); _r, _o = _x["pred"]["ridge"], _x["pred"]["dom20"]
        _rows.append([_tag, _op.get("knn"), _op.get("cut_um"), _op.get("lazy"), _n, _x["sigma_um"]["1"], _r["pcc"], _o["pcc"], _r["band_pcc"], _o["band_pcc"], _x["rel_gap"]["dom20"]["overall"], _x["rel_gap"]["dom20"]["fineband"],
                      _r["band_pcc_q25"], _o["band_pcc_q25"], _r["band_pcc_wvar"], _o["band_pcc_wvar"], _r["band_pcc_thr5"], _o["band_pcc_thr5"], _r["n_genes_thr5"]])
sheet("Operator_sensitivity", ["setting", "knn", "cut_um", "lazy", "region", "sigma1_um", "pcc_ridge", "pcc_oracle", "beta1_ridge", "beta1_oracle", "delta_scalar", "delta_fine",
                               "beta1_ridge_q25", "beta1_oracle_q25", "beta1_ridge_wvar", "beta1_oracle_wvar", "beta1_ridge_thr5", "beta1_oracle_thr5", "n_genes_thr5"],
      _rows, "Appendix F, Table operator", "results/blocks_xen_bands_{base,k4,k12,lazy75}/*.json", "src/blocks_xen_bands.py --knn/--cut_um/--lazy; base reproduces blocks_xen_bands")
_rows = []
for _n, _c in _load_dir("xen_band_ceiling").items():
    for _cp in _c["cps"]:
        _rows.append([_n, _c["n_bins"], _c["n_genes"], _c["reps"], _cp, _c["c_half"][str(_cp)], _c["c_full"][str(_cp)], float(np.sqrt(max(_c["c_full"][str(_cp)], 0)))])
    _rows.append([_n, _c["n_bins"], _c["n_genes"], _c["reps"], "scalar", _c["scalar_c_half"], _c["scalar_c_full"], float(np.sqrt(max(_c["scalar_c_full"], 0)))])
sheet("Band_reliability", ["region", "n_bins", "n_genes", "reps", "t", "c_half", "c_full (Spearman-Brown)", "sqrt(c_full) ceiling"], _rows, "Appendix I.4, Table reliab", "results/xen_band_ceiling/*.json", "xen_band_ceiling.py: binomial split-half of counts, same operator and bands")
_rows = []
for _f in sorted(glob.glob(os.path.join(R, "hest_controls_*.json"))):
    if "summary" in _f: continue
    _e = os.path.basename(_f)[len("hest_controls_"):-5]; _C = json.load(open(_f))["samples"]; _B = J(f"hest_blocks_{_e}.json")["samples"]; _P = J(f"hest_effres_ps_{_e}.json")["per_sample_pcc"]; _S = J(f"hest_rsel_ps_{_e}.json")["per_sample_pcc"]
    for _s, _v in _C.items():
        _rows.append([_e, _s, _v["cohort"], _v["n"], _P.get(_s), _S.get(_s), _B[_s]["blk_k20"], _v["oracle_image"], _v["oracle_coord"], _v["oracle_random"], _v["trainonly_image"], _v.get("n_train_samples"), _v.get("clusters_unseen_in_train")])
sheet("Controls_HEST", ["encoder", "sample", "cohort", "n_spots", "pcc_ridge_official", "pcc_ridge_cohort_selected", "oracle_image_published (blk_k20)", "oracle_image_recomputed", "oracle_coordinate", "oracle_random_matched", "trainonly_image", "n_train_samples", "clusters_unseen_in_train"],
      _rows, "Appendix I.5, Table controls_hest", "results/hest_controls_*.json, hest_blocks_*.json, hest_effres_ps_*.json", "hest_controls.py; published oracle used in the paper")
_rows = []; _drows = []
for _n, _d in _load_dir("xen_controls").items():
    for _K in ("K20", "K200"):
        _k = _d[_K]; _rows.append([_n, _d["n"], _K[1:], _d["ridge"]["pcc"], _k["oracle_image"]["pcc"], _k["oracle_coord"]["pcc"], _k["oracle_random_matched"]["pcc"], _k["trainonly_image"]["pcc"], _k["trainonly_coord"]["pcc"]])
    for _nm, _v in _d["degradations"].items():
        _drows.append([_n, _nm, _v["strength"], _v["pcc"], _v["flag"], _d["ridge"]["pcc"], _v["bands"]["1"], _d["K20"]["oracle_image"]["bands"]["1"]] + [_v["bands"][str(c)] for c in (2, 4, 8, 16, 32)])
sheet("Controls_Xenium", ["region", "n_bins", "K", "pcc_ridge", "oracle_image", "oracle_coordinate", "oracle_random_matched", "trainonly_image", "trainonly_coordinate"], _rows, "Appendix I.5, Table controls_xen", "results/xen_controls/*.json", "xen_controls.py; same 16x16 block CV as the ridge")
sheet("Degradations", ["region", "degradation", "strength", "pcc_after", "match_flag", "pcc_ridge_target", "beta1", "beta1_oracle_K20", "beta_t2", "beta_t4", "beta_t8", "beta_t16", "beta_t32"], _drows, "Appendix I.5, Table degrade", "results/xen_controls/*.json", "strength bisected to the ridge's scalar PCC (tol 0.004); hotspot removal hi-limited")
_rows = []
_orig = J("thitogene_hest.json")
for _run, _m in [("original", _orig)] + [(f"seed{sd}", {k: v for f in glob.glob(os.path.join(R, "thito_seeds", f"thitogene_*_s{sd}.json")) for k, v in json.load(open(f)).items()}) for sd in (1, 2)]:
    for _s, _v in _m.items(): _rows.append([_run, _s, _v["cohort"], _v["pcc"]])
sheet("THItoGene_runs", ["run", "sample", "cohort", "pcc"], _rows, "Appendix G", "results/thitogene_hest.json, results/thito_seeds/*.json", "original = unseeded first run (Table 7); seed1/seed2 via --seed")
if os.path.exists(os.path.join(R, "hest_octave_delta.json")):
    _o = J("hest_octave_delta.json")
    sheet("HEST_octave_delta", ["encoder", "n_cohorts", "delta_scalar", "delta_fine", "cohorts_oracle_above_scalar", "cohorts_oracle_above_fine", "cohorts_fine_gap_larger", "beta1_ridge", "beta1_oracle", "pcc_ridge", "pcc_oracle", "sigma1_um", "pitch_um"],
          [[d["enc"], d["n_coh"], d["ds"], d["df"], d["oracle_above_scalar"], d["oracle_above_fine"], d["fine_gap_larger"], d["b1r"], d["b1o"], d["pr"], d["po"], d["sig1"], d["pitch"]] for d in _o],
          "Appendix (benchmark-scale OCTAVE)", "results/hest_octave_delta.json", "scripts/hest_octave_delta.py; ridge official alpha (hest_effres_ps[_full]), oracle bands (hest_oracle_bands.py)")
if os.path.exists(os.path.join(R, "xen_multi_rank.json")):
    _m = J("xen_multi_rank.json"); _rows = []
    for _e, _v in _m["encoders"].items(): _rows.append([_e, _v["pcc"], _v["oracle_pcc"], _v["ratio"], _v["gap_ratio"], _v["sigma"]] + [_v["bands"][c] for c in sorted(_v["bands"], key=int)])
    _cps = sorted(next(iter(_m["encoders"].values()))["bands"], key=int)
    sheet("Xenium_multi_encoder", ["encoder", "pcc_ridge", "pcc_oracle_K20", "oracle_over_ridge", "fine_over_scalar_gap", "sigma1_um"] + [f"beta_t{c}" for c in _cps], _rows, "Appendix (multi-encoder OCTAVE)", "results/xen_multi_rank.json", "scripts/xen_multi_rank.py; specimen medians")


# ── 第四份审稿意见的补充分析 ──
_rows = []
for _n, _d in _load_dir("xen_attrib").items():
    _m, _c, _t, _g, _o = _d["model_decomp"], _d["crossfit"], _d["trainonly"], _d["contiguity"], _d["operators"]
    _rows.append([_n, _d["n"], _d["n_ok"], _m["pcc_model"], _m["pcc_model_between_only"], _m["pcc_between_vs_between"], _m["pcc_within_vs_within"], _m["pcc_oracle"], _m["beta1_model"], _m["beta1_model_between_only"], _m["beta1_within_vs_within"], _m["beta1_oracle"], _m["var_share_within_model"], _m["var_share_within_truth"],
                  _c["pcc_ridge"], _c["pcc_oracle_cf"], _c["pcc_oracle_same"], _c["beta1_ridge"], _c["beta1_oracle_cf"], _c["beta1_oracle_same"], _c["ratio_cf"], _c["ratio_same"], _t["pcc"], _t["beta1"], _t["ratio"],
                  _g["image_partition"]["neighbour_agreement"], _g["image_partition"]["components_per_cluster_median"], _g["image_partition"]["largest_component_share_median"], _g["coordinate_partition"]["neighbour_agreement"], _g["random_partition"]["neighbour_agreement"],
                  _o["lazy_rw_default"]["ratio"], _o["symmetric_normalised"]["ratio"], _o["gaussian_16um"]["ratio"]])
if _rows:
    sheet("Attribution_Xenium", ["region", "n_bins", "n_ok", "pcc_model", "pcc_model_between_only", "pcc_between_vs_between", "pcc_within_vs_within", "pcc_oracle", "beta1_model", "beta1_model_between", "beta1_within_vs_within", "beta1_oracle", "var_share_within_model", "var_share_within_truth",
                                 "cf_pcc_ridge_halfB", "cf_pcc_oracle_crossfit", "cf_pcc_oracle_samehalf", "cf_beta1_ridge", "cf_beta1_oracle_crossfit", "cf_beta1_oracle_samehalf", "ratio_crossfit", "ratio_samehalf", "trainonly_pcc", "trainonly_beta1", "trainonly_ratio",
                                 "img_neighbour_agreement", "img_components_per_cluster_median", "img_largest_component_share", "coord_neighbour_agreement", "random_neighbour_agreement", "ratio_lazy_default", "ratio_symmetric", "ratio_gaussian16"],
          _rows, "Appendix I.5 (attribution), Table attrib", "results/xen_attrib/*.json", "xen_attrib.py: model-side decomposition, cross-fitted oracle, train-only bands, contiguity, alternative operators")
_rows = []
for _f in sorted(glob.glob(os.path.join(R, "multi_ds", "*.json"))):
    _d = json.load(open(_f)); _r = _d["readouts"]
    _rows.append([_d["tower"], _d["name"], _d["n_ok"], _d["scores"]["pcc"], _d["scores"]["beta1"]] + [_r.get(k) for k in ["svg_top_jaccard", "svg_rank_rho", "hotspot_jaccard", "boundary_shift_um", "coloc_preserve", "hotspot_recall_selectivity"]])
if _rows:
    sheet("Downstream_11_encoders", ["encoder", "region", "n_ok", "pcc", "beta1", "svg_top_jaccard", "svg_rank_rho", "hotspot_jaccard", "boundary_shift_um", "coloc_preserve", "hotspot_recall_selectivity"], _rows, "Appendix (downstream utility)", "results/multi_ds/*.json", "multi_downstream.py; readouts from oracle_downstream.readouts")


_rows = [[_d["name"], _d["n"], _d["n_ok"], _d["pcc"], _d["between_term"], _d["within_term"], _d["between_share"], _d["within_share"], _d["check_max_abs_err"], _d["isolated_nodes"], _d["min_degree"]] for _n, _d in _load_dir("xen_addsplit").items()]
if _rows:
    sheet("Additive_PCC_split", ["region", "n_bins", "n_ok", "pcc", "between_term", "within_term", "between_share", "within_share", "max_abs_err", "isolated_nodes_29um", "min_degree"], _rows, "Appendix I.5 (additive split), Appendix B (isolated nodes)", "results/xen_addsplit/*.json", "xen_addsplit.py: common-denominator split of per-gene PCC; isolated bins in the 8-NN/29 um graph")

_rows = []
for _f in sorted(glob.glob("results/istar_xen/*.json")):
    _d = json.load(open(_f)); _h = json.load(open(_f.replace("istar_xen/", "istar_xen_hipt/"))) if os.path.exists(_f.replace("istar_xen/", "istar_xen_hipt/")) else None
    _rows.append([_d["name"], _d.get("n_test"), _d["istar"]["pcc"], _d["istar"]["beta1"], _d["istar"].get("fine_var_share"),
                  _h["ridge_hipt_official"]["pcc"] if _h else None, _h["ridge_hipt_official"]["beta1"] if _h else None, _h["ridge_hipt_1e4"]["pcc"] if _h else None, _h["ridge_hipt_1e4"]["beta1"] if _h else None,
                  _d["ridge"]["pcc"], _d["ridge"]["beta1"], _d["ridge"].get("fine_var_share"), _d["context_ridge"]["pcc"], _d["context_ridge"]["beta1"], _d["trainonly"]["pcc"], _d["trainonly"]["beta1"], _d["oracle"]["pcc"], _d["oracle"]["beta1"], _d.get("truth_fine_share")])
if _rows:
    sheet("iStar_Xenium", ["region", "n_test_bins", "istar_pcc", "istar_beta1", "istar_fine_share", "ridge_hipt_pcc", "ridge_hipt_beta1", "ridge_hipt_1e4_pcc", "ridge_hipt_1e4_beta1", "ridge_pcc", "ridge_beta1", "ridge_fine_share", "context_ridge_pcc", "context_ridge_beta1", "trainonly_pcc", "trainonly_beta1", "oracle_pcc", "oracle_beta1", "truth_fine_share"], _rows, "Appendix table tab:istar", "results/istar_xen/*.json, results/istar_xen_hipt/*.json", "istar_eval_xen.py, istar_hipt_ridge_xen.py; half split, 200 genes, same bins and operator")

if os.path.exists("results/beta1_rank_boot.json"):
    _b = json.load(open("results/beta1_rank_boot.json"))
    sheet("Beta1_rank_bootstrap", ["n_encoders", "n_pairs", "n_reversed_pairs", "reversals_stable_95pct", "reversals_stable_80pct", "median_pair_stability", "spearman_boot_vs_full_median", "spearman_lo", "spearman_hi", "beta1_ci_width_median", "beta1_ci_width_max", "keep_minus_dinov3h_point", "keep_minus_dinov3h_lo", "keep_minus_dinov3h_hi", "top1_stable_frac", "B"],
          [[_b["n_enc"], _b["n_pairs"], _b["n_rev"], _b["rev_stable95"], _b["rev_stable80"], _b["rev_stable_median"], _b["spearman_boot_vs_full"]["med"], _b["spearman_boot_vs_full"]["lo"], _b["spearman_boot_vs_full"]["hi"], _b["b1_ci_width"]["med"], _b["b1_ci_width"]["max"], _b["keep_minus_dinov3h"]["point"], _b["keep_minus_dinov3h"]["lo"], _b["keep_minus_dinov3h"]["hi"], _b["top1_stable"], _b["B"]]],
          "Appendix J.6 (bootstrap sentence)", "results/beta1_rank_boot.json", "scripts/beta1_rank_boot.py; specimen-level bootstrap of the 57-encoder finest-band ranking")

ws = wb.create_sheet("README", 0)
ws.append(["Source data for all figures and tables"])
ws["A1"].font = Font(bold=True, size=13)
ws.append([]); ws.append(["Sheet", "Figure or table", "Rows", "Source file", "Note"])
for c in ws[3]:
    c.font = Font(bold=True); c.fill = HEAD
for r in IDX: ws.append(list(r))
ws.append([])
ws.append(["Every value is read from a frozen JSON under results/. No value is typed by hand."])
ws.append(["Regenerate with: python3 scripts/mk_sourcedata_xlsx.py"])
for i, w in enumerate([26, 20, 8, 46, 64], 1): ws.column_dimensions[get_column_letter(i)].width = w
for row in ws.iter_rows(min_row=4):
    row[4].alignment = Alignment(wrap_text=True, vertical="top")
ws.freeze_panes = "A4"

wb.save(OUT)
print(f"\n{len(IDX)} 张工作表 -> {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB)")
