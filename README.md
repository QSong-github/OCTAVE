# Octave

Code for **Octave: Scale-Resolved Evaluation of Spatial Gene Expression Prediction from Histology**
(manuscript under review).

Octave evaluates histology-to-spatial-transcriptomics prediction at the scales the measurement
actually resolves. It provides (i) a *domain oracle*, the best predictor that only recognises
image domains, built from a frozen encoder's own features; (ii) the calibration of a correlation
score into an effective resolution, and the proof that within one fold this is a monotone
reparameterisation of the score; and (iii) the band-resolved correlation over dyadic diffusion
scales that gives the method its name.

## Installation

```bash
conda env create -f environment/environment_hest.yml     # PyTorch, timm, transformers, open_clip, trident
conda env create -f environment/environment_tfpf.yml     # TensorFlow 2.17, only for Google Path Foundation
```

The main environment runs every encoder except Path Foundation, whose released weights are a
TensorFlow SavedModel. Gated encoders (for example UNI, Virchow, H-optimus, GigaPath, mSTAR,
Pathryoshka) require an approved Hugging Face token; place it where `huggingface_hub` expects it.

## Data

Three sources. None is redistributed here.

| Dataset | What we use | How to obtain |
|---|---|---|
| HEST-Benchmark | 72 samples, 10 cohorts, official splits and 50-gene panels | Gated on Hugging Face under CC BY-NC-SA 4.0: accept the terms, then point `B` in `src/hest_embed_v2.py` and `src/hest_effres_ps.py` at `bench_data/` |
| Xenium | 16 regions from 8 specimens, binned to 16 µm | 10x Genomics public datasets, under their terms: `src/fetch_xenium.sh`, then `src/xen_prep.py` |
| Visium HD | Two colon sections at 16 µm | 10x Genomics public datasets, under their terms: `src/fetch_hd.sh`, `src/fetch_p5.sh`, then `src/hd_prep.py` |

## Models

The evaluation scores 57 frozen image encoders paired with a ridge head. Six published
end-to-end methods are run separately, as a check that the ridge baseline is not weak.

| Models | What we use | How to obtain |
|---|---|---|
| Frozen encoders | 57 pathology and general-purpose encoders, 6M to 1.1B parameters, each with its own published preprocessing | Named and configured one by one in `src/hest_embed_v2.py`, which loads each from its published source through `timm`, `transformers`, `open_clip` or the authors' own loader. UNI, Virchow, GigaPath, H-optimus, CONCH and others are gated: request access on Hugging Face and leave the token where `huggingface_hub` finds it. Path Foundation is a TensorFlow SavedModel and runs in the second environment |
| Published methods | HisToGene, Hist2ST, BLEEP, HECLIP, HGGEP, THItoGene, and iStar on the Xenium protocol | Clone the authors' repositories into `methods/`; the adaptors `src/<method>_hest.py` run them under the benchmark protocol and document every change |

## Usage

**Benchmark: domain oracle against a trained model, per encoder.**

```bash
# 1. embeddings for one encoder (GPU); 57 encoder names are listed in src/hest_embed_v2.py
python src/hest_embed_v2.py --encoder hoptimus0 --batch 128
# 2. ridge at the benchmark's alpha, an alpha grid, and the domain oracle at K = 20, 50, 200 (CPU)
python src/hest_effres_ps.py --encoder hoptimus0 --skip_sigma --out results/hest_effres_ps_hoptimus0.json
BLK_K=20,50,200 python pipeline/hest_blocks.py hoptimus0        # add BLK_ZSCORE=1 to standardise features first
# 3. leave-one-cohort-out alpha, then the aggregate over encoders
python pipeline/ridge_loco.py hoptimus0
python pipeline/blk_agg.py && python pipeline/blkz_agg.py
# 4. tables and headline numbers from the result files
python scripts/mk_tables.py && python scripts/headline_numbers.py
```

`pipeline/drv_one.sh <encoder ...>` chains these stages on SLURM. Published methods are run fold by
fold with `src/<method>_hest.py`, merged with `pipeline/merge_folds.py` and compared with
`pipeline/methods_agg2.py`.

**Xenium and Visium HD: calibration curves, band scores and shortfalls.**

```bash
python src/blocks_xen_bands.py <region>          # per region: calibration curve, band scores, shortfalls
python pipeline/blocks_xen_agg.py                # aggregate to specimens
python pipeline/split_sens.py                    # spatial block cross-validation sweep
python pipeline/band_homog.py                    # band homogeneity across samples
python pipeline/make_figs_new.py 3               # figures (run with 32 threads; a reproducibility guard aborts on drift)
```

**Controls, attribution and comparisons.** Each of these keeps the protocol of the corresponding
main run and changes one ingredient; the aggregators write the tables and the appendix text.

```bash
# partition controls: coordinate, random matched-size, means learned from training data
python pipeline/hest_controls.py --encoder <enc>            # benchmark
python pipeline/xen_controls.py --name <region>             # Xenium
# inductive partition: PCA and K-means fitted on training data only, test spots assigned to centres
python pipeline/hest_inductive.py --encoder <enc>
python pipeline/xen_inductive.py --name <region>            # also fits ridge with spatial context
# split the model's own prediction into between- and within-cluster parts (no test label)
python pipeline/xen_attrib.py --name <region>
python pipeline/xen_addsplit.py --name <region>             # exact additive split of the correlation
# the benchmark resolved by scale, and the noise ceiling per band
python pipeline/hest_oracle_bands.py --encoder <enc>
python pipeline/xen_band_ceiling.py --name <region>         # binomial thinning, Spearman-Brown
# an end-to-end super-resolution method under the same protocol, and its backbone-matched control
python pipeline/istar_prep_xen.py --name <region>           # then jobs/istar_xen_gpu.sh, istar_eval_xen.py
python pipeline/istar_hipt_ridge_xen.py --name <region>     # ridge on iStar's own HIPT features
# all encoders on Xenium: embeddings, then the band decomposition per encoder and region
python pipeline/xen_embed_all.py --encoder <enc> --name <region>
python pipeline/xen_pf_cut.py ... && python pipeline/xen_pf_embed_shard.py ...   # Path Foundation, two stages
python scripts/xen_multi_rank.py && python scripts/beta1_rank_boot.py            # ranking and its bootstrap
```

## Repository structure

```
src/          pipeline modules: encoder loading and embedding, ridge and effective resolution on the
              benchmark, the Octave pipeline on Xenium and Visium HD, method adaptors, data preparation
pipeline/     top-level drivers and aggregation: domain oracle, leave-one-cohort-out ridge, method
              comparison, parameter counts, probes, figure scripts, SLURM chain drivers (drv_*.sh)
jobs/         SLURM job scripts (paths point at our cluster's project directory; edit for your cluster)
scripts/      table, text and source-data generators: every number in the manuscript is written by one
              of these from a result file, so no figure or table is typed by hand
environment/  conda environment exports
```

## Conventions

- Every encoder uses its own published preprocessing (mean/std, input size, pooling); the choice is
  recorded next to each entry in `src/hest_embed_v2.py`, and a CPU probe (`pipeline/probe_cfg.py`)
  checks it before any full run.
- Statistics are computed at the specimen level (Xenium) or cohort level (benchmark) with exact
  two-sided sign tests.
- Embeddings, weights, data and logs are excluded by `.gitignore` and never committed.

## Citation

The manuscript is under review. A citation entry will be added when it is public.

## License

The code in this repository is released under the MIT License (see `LICENSE`). The data it reads
are not ours to relicense: HEST-Benchmark is CC BY-NC-SA 4.0 and gated, the Xenium and Visium HD
datasets follow 10x Genomics' terms, several encoder weights are gated by their publishers, and
each published method keeps the licence of its own repository.
