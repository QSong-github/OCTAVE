# Octave

Code for **Octave: Scale-Resolved Evaluation of Spatial Gene Expression Prediction from Histology**
(manuscript under review). This repository contains the evaluation pipeline only; the manuscript,
figures and result files are not distributed here.

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

| Source | Use | How to obtain |
|---|---|---|
| HEST-Benchmark | 72 samples, 10 cohorts, official splits and 50-gene panels | Download from the HEST release and point `B` in `src/hest_embed_v2.py` and `src/hest_effres_ps.py` at `bench_data/` |
| Xenium | 16 regions from 8 specimens, binned to 16 µm | `src/fetch_xenium.sh`, then `src/xen_prep.py` |
| Visium HD | Two colon sections at 16 µm | `src/fetch_hd.sh`, `src/fetch_p5.sh`, then `src/hd_prep.py` |
| Published methods | HisToGene, Hist2ST, BLEEP, HECLIP, HGGEP, THItoGene | Clone the authors' repositories into `methods/`; the adaptors `src/<method>_hest.py` run them under the benchmark protocol and document every change |

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

## Repository structure

```
src/          pipeline modules: encoder loading and embedding, ridge and effective resolution on the
              benchmark, the Octave pipeline on Xenium and Visium HD, method adaptors, data preparation
pipeline/     top-level drivers and aggregation: domain oracle, leave-one-cohort-out ridge, method
              comparison, parameter counts, probes, figure scripts, SLURM chain drivers (drv_*.sh)
jobs/         SLURM job scripts (paths point at our HiPerGator project directory; edit for your cluster)
scripts/      table, source-data and headline-number generators; scripts/src/ is the frozen copy of the
              modules as they were when the manuscript's numbers were computed
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

To be added by the authors before public release.
