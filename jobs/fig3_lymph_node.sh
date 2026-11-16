#!/bin/bash
#SBATCH -J fig3
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 2 --mem=16G -t 2:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
# marker-gene maps on the Xenium lymph node (measured, domain oracle, Hibou-L, CONCH v1, KEEP).
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project; export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
R=Xenium_Prime_Human_Lymph_Node_Reactive_FFPE
for E in conch_v1 keep; do [ -s results/xen_altrank_pred/${E}_${R}.npz ] || python -u xen_altrank.py --name $R --tower $E --save_pred; done
mkdir -p figures
python -u fig3_lymph_node.py
