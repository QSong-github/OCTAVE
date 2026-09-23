#!/bin/bash
#SBATCH -J tfenv
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 4 --mem=32G -t 2:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
conda create -y -q -n tfpf python=3.11 >/dev/null
conda activate tfpf
pip install -q "tensorflow[and-cuda]==2.17.*" "tf-keras==2.17.*" huggingface_hub h5py numpy pillow 2>&1 | tail -2
python - <<PY
import tensorflow as tf, tf_keras, huggingface_hub, h5py
print("tensorflow", tf.__version__, "| tf_keras", tf_keras.__version__, "| GPUs visible here:", len(tf.config.list_physical_devices("GPU")))
PY
