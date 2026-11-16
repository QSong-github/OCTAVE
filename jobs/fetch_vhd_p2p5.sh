#!/bin/bash
#SBATCH -J vhdget
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION --array=0-1 -c 1 --mem=4G -t 12:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
# re-fetch the two Visium HD colon sections of the paper (P2, P5) from the 10x CDN, as src/fetch_p5.sh does. Idempotent.
set -u
B=https://cf.10xgenomics.com/samples/spatial-exp/3.0.0
NAMES=(Visium_HD_Human_Colon_Cancer_P2 Visium_HD_Human_Colon_Cancer_P5)
N=${NAMES[$SLURM_ARRAY_TASK_ID]}
D=/path/to/project/data/visiumhd/$N; mkdir -p $D
echo "[$N] start $(date '+%F %T')"
[ -s $D/${N}_tissue_image.btf ] || curl -L --retry 5 --retry-delay 10 --max-time 21600 -o $D/${N}_tissue_image.btf $B/$N/${N}_tissue_image.btf
[ -d $D/binned_outputs/square_016um ] || curl -sL --retry 5 --retry-delay 10 --max-time 14400 $B/$N/${N}_binned_outputs.tar.gz | tar -xzf - -C $D --wildcards "*square_016um*" || true
ls -laR $D | head -40
echo "[$N] done $(date '+%F %T')"
