#!/bin/bash
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
B=https://cf.10xgenomics.com/samples/spatial-exp/3.0.0
N=Visium_HD_Human_Colon_Cancer_P5
D=/path/to/systema4ST/data/visiumhd/$N; mkdir -p $D
[ -s $D/${N}_tissue_image.btf ] || curl -L --retry 5 --max-time 21600 -o $D/${N}_tissue_image.btf $B/$N/${N}_tissue_image.btf
[ -d $D/binned_outputs/square_016um ] || curl -sL --retry 5 --max-time 14400 $B/$N/${N}_binned_outputs.tar.gz | tar -xzf - -C $D --wildcards "*square_016um*" || true
ls -la $D
