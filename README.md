# Octave

Code for **OCTAVE: A Scale-Resolved Benchmark for Spatial Gene Expression Prediction from Histology**.

Octave evaluates histology-to-spatial-transcriptomics prediction at the scales the measurement
actually resolves. It provides (i) the band score, the correlation between prediction and measurement in
each band of spatial scale over dyadic diffusion scales, which gives the method its name; (ii) a *domain
oracle*, the best predictor that only recognises image domains, built from a frozen encoder's own
features, as a reference; and (iii) a benchmark of 57 frozen image encoders and end-to-end methods on
Xenium, Visium HD and the HEST benchmark under one protocol and one summary rule.

## Installation

```bash
conda env create -f environment/environment_hest.yml     # PyTorch, timm, transformers, open_clip, trident
conda env create -f environment/environment_tfpf.yml     # TensorFlow 2.17, only for Google Path Foundation
```

The main environment runs every encoder except Path Foundation, whose released weights are a
TensorFlow SavedModel. Gated encoders (for example UNI, Virchow, H-optimus, GigaPath, mSTAR,
Pathryoshka) require an approved Hugging Face token; place it where `huggingface_hub` expects it.

## Data

We redistribute nothing. The scripts below download from the owners' own servers and from
nowhere else, and each source keeps its own licence and access conditions.

| Dataset | What we use | Where it comes from |
|---|---|---|
| HEST-Benchmark | 72 samples, 10 cohorts, official splits and 50-gene panels | `huggingface.co/datasets/MahmoodLab/hest`, CC BY-NC-SA 4.0 and gated: make an account, accept the terms, download, then point `B` in `src/hest_embed_v2.py` and `src/hest_effres_ps.py` at your `bench_data/`. TRIPLEX also needs the whole-slide images (`jobs/fetch_hest_wsis.sh`, `jobs/fetch_hest_st.sh`, `jobs/fetch_hest_seg.sh`) |
| Xenium | 22 regions from 13 specimens, binned to 16 µm | 10x Genomics public datasets (`www.10xgenomics.com/datasets`), files served from `cf.10xgenomics.com/samples/xenium/`. `jobs/fetch_xenium22.sh` names the 22 regions, pulls the H\&E image, its alignment matrix and the cell-feature matrix, and bins them to 16 µm with `src/xen_prep.py` |
| Visium HD | Two colon sections at 16 µm | Same source, files served from `cf.10xgenomics.com/samples/spatial-exp/3.0.0/`. `jobs/fetch_vhd_p2p5.sh` pulls the 2 sections, and `jobs/vhd_prep2.sh` writes the pyramidal H\&E and the 16 µm bins (`src/hd_prep.py`, `src/rebuild_binned16.py`) |

## Models

The evaluation scores 57 frozen image encoders paired with a ridge head. Eight end-to-end
methods are run separately on the benchmark, and iStar on the Xenium protocol, as a check that
the ridge baseline is not weak.

| Models | What we use | How to obtain |
|---|---|---|
| Frozen encoders | 57 pathology and general-purpose encoders, 6M to 1.1B parameters, each with the preprocessing its authors specify | Named and configured one by one in `src/hest_embed_v2.py`, which loads each from its published source through `timm`, `transformers`, `open_clip` or the authors' own loader. UNI, Virchow, GigaPath, H-optimus, CONCH and others are gated: request access on Hugging Face and leave the token where `huggingface_hub` finds it. Path Foundation is a TensorFlow SavedModel and runs in the second environment |
| End-to-end methods | HisToGene, Hist2ST, BLEEP, HECLIP, HGGEP, THItoGene, DeepSpot and TRIPLEX on the benchmark; iStar on the Xenium protocol | Clone the authors' repositories into `methods/`; the adaptors `src/<method>_hest.py` run them under the benchmark protocol and document every change. DeepSpot takes its features from `src/deepspot_feats.py`; TRIPLEX runs through the authors' code after `src/triplex_subsample.py` and the neighbourhood dumps of `jobs/triplex_hest.sh` |

## Usage

Run every command from the project root. Result files go to `results/` (or `$OCTAVE_RESULTS`), and the
table and figure scripts write into `paper/` unless a paper directory is given as their argument. The
`jobs/` scripts run the same steps on SLURM.

**HEST benchmark: frozen encoders, domain oracle and controls.**

```bash
# 1. embeddings for one encoder (GPU); the 57 encoder names are listed in src/hest_embed_v2.py
python src/hest_embed_v2.py --encoder hoptimus0 --batch 128
# 2. per-sample ridge and band scores, the domain oracle at K = 20, 50, 200, and the partition controls (CPU)
python src/hest_effres_ps.py --encoder hoptimus0 --skip_sigma --out results/hest_effres_ps_hoptimus0.json
BLK_K=20,50,200 python pipeline/hest_blocks.py hoptimus0        # add BLK_ZSCORE=1 to standardise features first
python pipeline/hest_controls.py --encoder hoptimus0
python pipeline/hest_inductive.py --encoder hoptimus0
python pipeline/hest_oracle_bands.py --encoder hoptimus0
python pipeline/hest_trainonly_coord.py                          # coordinate partition with training means
# 3. leave-one-cohort-out ridge penalty
python pipeline/ridge_loco.py hoptimus0
# 4. every HEST summary under the summary rule (see Conventions), then the HEST tables
python scripts/hest_rule_numbers.py && python scripts/mk_tab_triplex.py && python scripts/mk_tab_controls.py
python scripts/mk_tab_hest_rule.py paper
python scripts/hest_octave_delta.py                              # widths of HEST's finest band
```

`pipeline/drv_one.sh <encoder ...>` chains these stages on SLURM. The end-to-end methods are run fold by
fold with `src/<method>_hest.py` and merged with `pipeline/merge_folds.py`; `jobs/cp10k_rsel.sh` repeats
the ridge models under a second expression normalisation.

**Xenium: band scores, all encoders, controls.**

```bash
python src/blocks_xen_bands.py --name <region>                   # ridge on Hibou-L and the domain oracle, every band
python pipeline/xen_embed_all.py --encoder <enc> --h5ad <bins.h5ad> --tiff <he.tif> --out <emb.npy>   # other encoders (jobs/xen_emb_all.sh); then blocks_xen_bands.py --tower <enc>
python pipeline/xen_pf_cut.py ... && python pipeline/xen_pf_embed_shard.py ...   # Path Foundation, two stages
python scripts/xen_multi_rank.py && python scripts/beta1_rank_boot.py && python scripts/mk_tab_multi.py
python src/blocks_xen_seeds.py --name <region> --nseed 16 && python src/blocks_xen_ablate.py --name <region>   # K-means seeds, one setting at a time
python pipeline/xen_controls.py --name <region> && python pipeline/xen_inductive.py --name <region>
python pipeline/xen_ctrl57.py --name <region> && python pipeline/agg_xen_ctrl57.py   # controls for all 57 encoders
python pipeline/xen_attrib.py --name <region> && python pipeline/xen_addsplit.py --name <region>
python pipeline/xen_band_ceiling.py --name <region>              # split-half reliability per band
python pipeline/istar_prep_xen.py --name <region>                # then jobs/istar_xen_gpu.sh and pipeline/istar_eval_xen.py
python pipeline/istar_hipt_ridge_xen.py --name <region>          # ridge on iStar's own HIPT features
python pipeline/xen_altmetrics.py --name <region> && python pipeline/xen_mssi.py --name <region>   # MSSI (separate env, jobs/mk_env_mssi.sh)
python pipeline/grid_table.py                                    # scoring on the prediction's own grid and on a fixed grid
```

**Visium HD.** `jobs/vhd_emb.sh` embeds both sections with every encoder, `jobs/vhd_bands.sh` runs
`src/blocks_vhd_bands.py`, `jobs/vhd_ctrl57.sh` and `jobs/vhd_ind57.sh` run `pipeline/vhd_ctrl57.py`, and
`pipeline/agg_vhd_ctrl57.py` and `scripts/vhd57_bench.py` summarise them. `jobs/vhd_repro2.sh` and
`src/vhd_repro_run.py` reproduce the fold and window analyses, and `jobs/protocol_choices.sh` the effect of
the gene panel and the exclusion buffer.

**Tables and figures of the paper, in this order.**

```bash
python scripts/mk_tab_controls_main.py paper                     # Table 1 (3 datasets)
python scripts/supp_rule_b.py paper                              # supplementary Xenium tables and numbers
python scripts/mk_octave_bench.py paper && python scripts/mk_tab_bench57.py paper   # Table 2
python scripts/mk_attrib.py && python scripts/istar_agg.py       # supplementary attribution and iStar tables
OCTAVE_ROOT=. OUT=paper/figs python scripts/fig2_scale.py        # Figure 2
python pipeline/fig3_lymph_node.py                               # Figure 3 (see jobs/fig3_lymph_node.sh)
python scripts/fig4_bench.py paper && python scripts/fig5_encoders.py paper   # Figures 4 and 5
python src/fig1_framework_data.py && python src/fig1_v3_data.py  # Figure 1 data (jobs/fig1_framework.sh, jobs/fig1_v3.sh)
python scripts/fig1_v4.py paper/figs                             # Figure 1 layers: background, raster and labels
```

`scripts/fig1_v4.py` writes Figure 1 as three layers, which were combined in a vector editor; its encoder
icon is in `scripts/fig1_icons.py`. The window figure of the supplement is drawn by `pipeline/fig1_windows.py`
through `jobs/fig1_windows.sh`.

## Repository structure

```
src/          pipeline modules: encoder loading and embedding, ridge and effective resolution on the
              benchmark, the Octave pipeline on Xenium and Visium HD, method adaptors, data preparation
pipeline/     top-level drivers and aggregation: domain oracle, leave-one-cohort-out ridge, method
              comparison, parameter counts, probes, figure scripts, SLURM chain drivers (drv_*.sh)
jobs/         SLURM job scripts. Site-specific values are placeholders: `/path/to` for the project
              directory, `YOUR_QOS` for the QOS and `YOUR_CPU_PARTITION` / `YOUR_GPU_PARTITION` for the
              partitions. Set all four for your cluster before submitting anything.
scripts/      table and figure generators: every measured number in the manuscript is written by one of
              these from a result file; the one hand-typed table is the protocol-settings table
environment/  conda environment exports
```

The job scripts assume a flat project root: they `cd` into the project directory and call each module by
its bare file name (`python3 -u xen_ctrl57.py`). The release groups those same files into `src/` and
`pipeline/`, so copy or symlink both directories' contents into the project root before submitting, or
adjust the paths in the job scripts.

## Conventions

- Every encoder uses the preprocessing its authors specify (mean/std, input size, pooling); the choice is
  recorded next to each entry in `src/hest_embed_v2.py`, and a CPU probe (`pipeline/probe_cfg.py`)
  checks it before any full run.
- Sign tests are exact and two-sided, at the specimen level (Xenium) or the cohort level (HEST).
- Every number follows one summary rule. Each score is computed per sample (HEST) or per region
  (Xenium, Visium HD). It is summarised by its median over the 72 HEST samples, or within each Xenium
  specimen and then across the 13 specimens, with the 2 Visium HD sections in place of specimens.
  Across encoders the median is taken again. Every ratio, gap or difference is formed from these
  medians, and from the printed values when these are shown beside it. No summary uses a mean or a
  standard deviation. Cohorts and specimens are used to count how often one score is above another.
  `scripts/hest_rule_numbers.py` and `scripts/mk_tab_hest_rule.py` apply the rule to HEST, and
  `scripts/supp_rule_b.py` and `scripts/mk_tab_controls_main.py` to Xenium and Visium HD.
  `scripts/mk_tables.py` is kept only for its encoder name table; it writes older versions of the
  tables, so do not run it after the scripts above.
- Embeddings, weights, data and logs are excluded by `.gitignore` and never committed.

## License

The code in this repository is released under the MIT License (see `LICENSE`). The data it reads
are not ours to relicense: HEST-Benchmark is CC BY-NC-SA 4.0 and gated, the Xenium and Visium HD
datasets follow 10x Genomics' terms, several encoder weights are gated by their publishers, and
each end-to-end method keeps the licence of its own repository.
