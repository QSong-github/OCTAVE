# -*- coding: utf-8 -*-
"""Path Foundation 两段式第二段（tfpf 环境）：读 224 块分片，/255，serving_default → output_0，存 .npy。"""
import os, sys, argparse, numpy as np, h5py
TOK = open("/blue/qsong1/wang.qing/.cache/huggingface/token").read().strip(); os.environ["HF_TOKEN"] = TOK; os.environ["HUGGING_FACE_HUB_TOKEN"] = TOK
import tensorflow as tf
from huggingface_hub import snapshot_download
ap = argparse.ArgumentParser(); ap.add_argument("--h5", required=True); ap.add_argument("--out", required=True); ap.add_argument("--batch", type=int, default=128); a = ap.parse_args()
m = tf.saved_model.load(snapshot_download("google/path-foundation")); infer = m.signatures["serving_default"]; out = []
with h5py.File(a.h5, "r") as h:
    n = h["img"].shape[0]
    for i in range(0, n, a.batch):
        x = np.asarray(h["img"][i:i + a.batch]).astype(np.float32) / 255.0; out.append(infer(tf.constant(x))["output_0"].numpy().astype(np.float32))
E = np.concatenate(out); np.save(a.out, E); print(f"embedded {E.shape} -> {a.out}", flush=True)
