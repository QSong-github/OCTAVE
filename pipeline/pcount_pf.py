# -*- coding: utf-8 -*-
"""Path Foundation（TF SavedModel）的参数量：全部 variables 的元素数之和。"""
import os, json, numpy as np
TOK = open("/path/to/.cache/huggingface/token").read().strip()
os.environ["HF_TOKEN"] = TOK; os.environ["HUGGING_FACE_HUB_TOKEN"] = TOK
import tensorflow as tf
from huggingface_hub import snapshot_download
m = tf.saved_model.load(snapshot_download("google/path-foundation"))
n = int(sum(int(np.prod(v.shape)) for v in m.variables))
print("  path_foundation %9.1f M  (%d variables)" % (n / 1e6, len(m.variables)))
P = "/path/to/systema4ST/results/encoder_params.json"
out = json.load(open(P)); out["path_foundation"] = {"params": n, "hidden": 384}
json.dump(out, open(P, "w"), indent=1); print("共 %d 个" % len(out))
