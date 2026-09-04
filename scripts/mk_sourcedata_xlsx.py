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
