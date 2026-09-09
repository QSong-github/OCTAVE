# -*- coding: utf-8 -*-
"""CPU 核对：Path Foundation SavedModel 能否加载、serving_default 对 224 输入给出 384 维。"""
import os, numpy as np
TOK = open("/path/to/.cache/huggingface/token").read().strip()
os.environ["HF_TOKEN"] = TOK; os.environ["HUGGING_FACE_HUB_TOKEN"] = TOK
import tensorflow as tf
from huggingface_hub import snapshot_download
d = snapshot_download("google/path-foundation"); print("snapshot:", d, sorted(os.listdir(d)))
m = tf.saved_model.load(d); infer = m.signatures["serving_default"]
print("inputs:", infer.structured_input_signature); print("outputs:", infer.structured_outputs)
x = np.random.RandomState(0).rand(4, 224, 224, 3).astype(np.float32)
o = infer(tf.constant(x)); k = list(o)[0]; v = o[k].numpy()
print("key=%s shape=%s |mean|=%.4f std=%.4f nan=%d" % (k, v.shape, abs(v.mean()), v.std(), int(np.isnan(v).sum())))
assert v.shape == (4, 384) and not np.isnan(v).any()
print("PF_CHECK_OK")
