# -*- coding: utf-8 -*-
"""Google Path Foundation（ViT-S，TF-Keras 权重）在 HEST 72 个样本上抽嵌入。
输出与 src/hest_embed_v2.py 完全同格式：results/hest_emb/{sid}_path_foundation.npz，键 X (n,384) float32、bc (object)。
预处理照模型卡：224×224 RGB，除以 255 到 [0,1]，无均值方差归一化；signatures["serving_default"] → output_0。"""
import os, sys, glob, json, numpy as np, h5py, time
TOK = open("/path/to/.cache/huggingface/token").read().strip()
os.environ["HF_TOKEN"] = TOK; os.environ["HUGGING_FACE_HUB_TOKEN"] = TOK
import tensorflow as tf
from huggingface_hub import snapshot_download   # hub 1.x 已移除 from_pretrained_keras；仓库本身是 TF SavedModel
B = "/path/to/he2st/HEST/eval/bench_data"
EMB = "/path/to/systema4ST/results/hest_emb"
ENC = "path_foundation"; BS = 128
SHARD = int(os.environ.get("SHARD", 0)); NSHARD = int(os.environ.get("NSHARD", 1))
print("GPU:", tf.config.list_physical_devices("GPU"), flush=True)
m = tf.saved_model.load(snapshot_download("google/path-foundation")); infer = m.signatures["serving_default"]
os.makedirs(EMB, exist_ok=True); done = 0
for c in sorted(os.listdir(B)):
    pd_ = os.path.join(B, c, "patches")
    if not os.path.isdir(pd_): continue
    for fi, f in enumerate(sorted(glob.glob(os.path.join(pd_, "*.h5")))):
        if fi % NSHARD != SHARD: continue          # 数组分片：各任务处理互不重叠的文件
        sid = os.path.basename(f)[:-3]; out = os.path.join(EMB, f"{sid}_{ENC}.npz")
        if os.path.exists(out): done += 1; continue
        t0 = time.time()
        with h5py.File(f, "r") as h:
            bk = "barcodes" if "barcodes" in h else "barcode"
            bc = np.asarray(h[bk][:]).flatten().astype(str).tolist()
            n = h["img"].shape[0]; X = []
            for i in range(0, n, BS):
                x = np.asarray(h["img"][i:i + BS])[..., :3].astype(np.float32) / 255.0
                if x.shape[1:3] != (224, 224):
                    x = tf.image.resize(x, (224, 224), method="bilinear").numpy()
                o = infer(tf.constant(x))["output_0"].numpy()
                X.append(o.astype(np.float32))
        X = np.concatenate(X)
        np.savez(out, X=X, bc=np.array(bc, dtype=object)); done += 1
        print(f"  {c}/{sid}: X{X.shape}  {time.time()-t0:.0f}s", flush=True)
print(f"分片 {SHARD}/{NSHARD} 完成 {done} 个样本 -> {EMB}/*_{ENC}.npz")
