# -*- coding: utf-8 -*-
"""Google Path Foundation（TF SavedModel，CPU）在 Xenium 区域上抽嵌入，协议同 pathfound_embed.py：224×224 RGB /255，无均值方差归一化，
signatures['serving_default'] → output_0；切块同 hd_embed/xen_embed_all：openslide level 0，中心 obsm['pxl']，边长 ctx_px（61.4 µm 视野），再缩放到 224。"""
import os, sys, argparse, time, numpy as np, anndata as ad
TOK = open("/blue/qsong1/wang.qing/.cache/huggingface/token").read().strip(); os.environ["HF_TOKEN"] = TOK; os.environ["HUGGING_FACE_HUB_TOKEN"] = TOK
import tensorflow as tf
from huggingface_hub import snapshot_download
import openslide
ap = argparse.ArgumentParser(); ap.add_argument("--h5ad", required=True); ap.add_argument("--tiff", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ctx_px", type=int, default=224); ap.add_argument("--batch", type=int, default=128); a = ap.parse_args()
m = tf.saved_model.load(snapshot_download("google/path-foundation")); infer = m.signatures["serving_default"]
A = ad.read_h5ad(a.h5ad, backed="r"); pxl = np.asarray(A.obsm["pxl"], np.float64); sl = openslide.OpenSlide(a.tiff); half = a.ctx_px // 2; out = []; t0 = time.time()
print(f"[{os.path.basename(a.h5ad)}] {len(pxl)} bin ctx={a.ctx_px}", flush=True)
for i in range(0, len(pxl), a.batch):
    pb = pxl[i:i + a.batch]
    x = np.stack([np.asarray(sl.read_region((int(px - half), int(py - half)), 0, (a.ctx_px, a.ctx_px)).convert("RGB"), dtype=np.float32) for px, py in pb]) / 255.0
    if x.shape[1:3] != (224, 224): x = tf.image.resize(x, (224, 224), method="bilinear").numpy()
    out.append(infer(tf.constant(x))["output_0"].numpy().astype(np.float32))
    if (i // a.batch) % 100 == 0: print(f"    {i + len(pb)}/{len(pxl)}  {time.time()-t0:.0f}s", flush=True)
E = np.concatenate(out); os.makedirs(os.path.dirname(a.out), exist_ok=True); np.save(a.out, E); print(f"已存 {a.out} {E.shape} {time.time()-t0:.0f}s", flush=True)
