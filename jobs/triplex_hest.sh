#!/bin/bash
#SBATCH -J tpxhest
#SBATCH --qos=YOUR_QOS --partition=YOUR_GPU_PARTITION --gres=gpu:1 --array=0-19 -c 8 --mem=64G -t 36:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
# TRIPLEX 在 HEST 基准上：作者代码原样（script/01-preprocess_hest_bench.sh + train_hest.sh 的流程），只换数据源与配置路径。
# 两个主干：cigar（CVPR 论文所用 ResNet18 SSL，模型配置取 config/ST/andersson/TRIPLEX.yaml，emb_dim 512）；
#           uni_v1（仓库 HEST-bench 脚本的默认主干，配置取 config/GSE240429/TRIPLEX.yaml，emb_dim 1024）。训练超参取 GSE240429/default.yaml（Visium）。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate triplex
export PYTHONPATH=/path/to/project/methods/hest_new:/path/to/project/methods/triplex_shim:/path/to/project/methods/TRIPLEX/src
export WANDB_MODE=offline HF_HUB_OFFLINE=1 PYTHONWARNINGS=ignore TQDM_DISABLE=1
COH=(SKCM HCC LUNG PAAD COAD READ IDC LYMPH_IDC PRAD CCRCC); MOD=(cigar uni_v1)
I=$SLURM_ARRAY_TASK_ID; C=${COH[$((I % 10))]}; M=${MOD[$((I / 10))]}
B=/path/to/he2st/HEST/eval/bench_data; H=/path/to/project/data/hest_wsis; OUT=/path/to/project/data/triplex/$C
mkdir -p $OUT/splits; cp -n $B/$C/splits/*.csv $OUT/splits/; cp -n $B/$C/var_50genes.json $OUT/
python - "$OUT" <<'PY'
import sys, glob, csv, os
out=sys.argv[1]; ids=sorted({r["sample_id"] for f in glob.glob(f"{out}/splits/*.csv") for r in csv.DictReader(open(f))})
p=f"{out}/ids.csv"
if not os.path.exists(p): open(p,"w").write("sample_id\n"+"\n".join(ids)+"\n")
print("ids:", len(ids))
PY
NF=$(ls $OUT/splits | grep -c test_)
cd /path/to/project/methods/TRIPLEX
echo "[$C / $M] folds=$NF $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
flock $OUT/.prep.lock bash -c "[ -f $OUT/.prep_done ] || (python src/preprocess/prepare_data.py --output_dir $OUT --hest_dir $H --mode hest --slide_ext .tif && touch $OUT/.prep_done)"
[ -f $OUT/.prep_done ] || { echo "prepare_data 失败"; exit 1; }
flock $OUT/.prep.lock python /path/to/project/src/triplex_swap_coords.py $OUT/patches || { echo "坐标列交换失败"; exit 1; }
# 邻居 patch：作者 BC1/BC2/SCC 脚本的做法——prepare_data --save_neighbors 用 HEST 从 WSI 切 1120 px（0.5 µm/px）邻居块并按条码对齐到目标 patch；
# 仓库里 hest-bench 脚本直接从 WSI 逐块读的路径要求 patch h5 里没有 img，与 HEST 基准的 patch 文件不兼容（会拿 224 目标块当邻居块）。
# 邻居 patch：逐片判断是否齐全（.nbr_done 这类总标记会掩盖个别缺片），缺则补切；作者的 save_patches 见文件即跳过，天然增量。
MISSING() { for p in $OUT/patches/*.h5; do s=$(basename $p .h5); [ -f $OUT/patches/neighbor/$s.h5 ] || echo $s; done; }
flock $OUT/.prep.lock python /path/to/project/src/triplex_check_nbr.py $C
for s in $(MISSING); do rm -f $OUT/emb/neighbor/*/$s.h5; done          # 缺邻居块的片，其旧嵌入一并作废
if [ -n "$(MISSING)" ]; then echo "补切邻居块: $(MISSING | tr '\n' ' ')"; flock $OUT/.prep.lock python src/preprocess/prepare_data.py --output_dir $OUT --hest_dir $H --mode hest --slide_ext .tif --save_neighbors --num_n 5; fi
N_MISS=$(MISSING | wc -l); [ "$N_MISS" -eq 0 ] || { echo "邻居 patch 仍缺 $N_MISS 片: $(MISSING | tr '\n' ' ')"; exit 1; }
flock $OUT/.prep.lock python /path/to/project/src/triplex_subsample.py $C || { echo "下采样失败"; exit 1; }
flock $OUT/.prep.lock python /path/to/project/src/triplex_fix_adata.py $C || { echo "表达对齐失败"; exit 1; }
# 清理上次失败留下的空/坏嵌入文件（作者脚本见文件即跳过）
python - "$OUT" "$M" <<'PY2'
import sys, glob, os, h5py
out, m = sys.argv[1], sys.argv[2]; n = 0
for f in glob.glob(f"{out}/emb/*/{m}/*.h5"):
    try:
        with h5py.File(f, "r") as h: ok = "embeddings" in h and h["embeddings"].shape[0] > 0
    except Exception: ok = False
    if not ok: os.remove(f); n += 1
print("删除坏嵌入文件", n)
PY2
python src/preprocess/extract_img_features.py --wsi_dataroot $H/wsis --slide_ext .tif --patch_dataroot $OUT/patches --embed_dataroot $OUT/emb/global --num_n 1 --model_name $M --weights_root / --id_path $OUT/ids.csv --num_workers 2 || { echo "global 特征失败"; exit 1; }
python src/preprocess/extract_img_features.py --wsi_dataroot $H/wsis --patch_dataroot $OUT/patches/neighbor --embed_dataroot $OUT/emb/neighbor --slide_ext .tif --num_n 5 --model_name $M --weights_root / --id_path $OUT/ids.csv --batch_size 256 --num_workers 2 || { echo "neighbor 特征失败"; exit 1; }
python /path/to/project/src/triplex_align.py $C $M || { echo "对齐失败"; exit 1; }
n_g=$(ls $OUT/emb/global/$M 2>/dev/null | wc -l); n_n=$(ls $OUT/emb/neighbor/$M 2>/dev/null | wc -l); echo "特征: global $n_g, neighbor $n_n"
CFG=config/hest/${C}_${M}; mkdir -p $CFG
cat > $CFG/default.yaml <<YML
GENERAL:
  seed: 2021
  log_path: ./logs
  save_predictions: True

TRAINING:
  num_k: $NF
  learning_rate: 1.0e-4
  num_epochs: 200
  monitor: PearsonCorrCoef
  mode: max
  early_stopping:
    patience: 10
  lr_scheduler:
    patience: 5
    factor: 0.1
  save_best_only: True

DATA:
  data_dir: $OUT
  output_dir: $OUT/pred_$M
  gene_type: 'var'
  num_genes: 50
  num_outputs: 50
  cpm: False
  smooth: False
  model_name: '$M'

  train_dataloader:
        batch_size: 128
        num_workers: 4
        pin_memory: True
        shuffle: True

  test_dataloader:
      batch_size: 1
      num_workers: 4
      pin_memory: True
      shuffle: False
YML
if [ "$M" = "cigar" ]; then cp config/ST/andersson/TRIPLEX.yaml $CFG/TRIPLEX.yaml; else cp config/GSE240429/TRIPLEX.yaml $CFG/TRIPLEX.yaml; fi
# 是否需要训练：看有没有检查点（只看目录存在会被上一轮失败留下的空目录骗过，导致跳过训练、eval 无 ckpt 可读而静默产出 0 个预测）
NCKPT=$(find logs/hest/${C}_${M}/TRIPLEX -name "*.ckpt" 2>/dev/null | wc -l)
if [ "$NCKPT" -eq 0 ]; then rm -rf logs/hest/${C}_${M}/TRIPLEX; python src/main.py --config_name=hest/${C}_${M}/TRIPLEX --mode=cv --gpu=1 || { echo "训练失败"; exit 1; }; fi
find logs/hest/${C}_${M}/TRIPLEX -name "*.ckpt" | head -1 | grep -q . || { echo "训练未产出检查点"; exit 1; }
python src/main.py --config_name=hest/${C}_${M}/TRIPLEX --mode=eval --gpu=1 || { echo "eval 失败"; exit 1; }
NPRED=$(find $OUT/pred_$M -name "*.h5ad" 2>/dev/null | wc -l); echo "预测文件: $NPRED"; [ "$NPRED" -gt 0 ] || { echo "eval 未产出预测"; exit 1; }; ls logs/hest/${C}_${M}/TRIPLEX/*/fold0/eval 2>/dev/null | head -3
