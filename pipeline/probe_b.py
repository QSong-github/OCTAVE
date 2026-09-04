# -*- coding: utf-8 -*-
"""B 档三个（已授权）探针：hibou_b（trident）、gigapath-flash（timm）、Google Path Foundation（TF-Keras）。"""
import os, sys, json, warnings, numpy as np
warnings.filterwarnings("ignore")
TOK = open("/blue/qsong1/wang.qing/.cache/huggingface/token").read().strip()
os.environ["HF_TOKEN"] = TOK; os.environ["HUGGING_FACE_HUB_TOKEN"] = TOK
import torch, timm
from PIL import Image
dev = "cuda" if torch.cuda.is_available() else "cpu"
img = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
OK = {}
def report(name, f):
    try:
        dim, note = f(); OK[name] = {"dim": int(dim), "note": note}; print(f"  ✓ {name:<16s} dim={dim:<5d} {note}", flush=True)
    except Exception as e: print(f"  ✗ {name:<16s} {type(e).__name__}: {str(e)[:140]}", flush=True)
def hibou_b():
    from trident.patch_encoder_models.load import encoder_factory
    m = encoder_factory("hibou_b").eval().to(dev)
    x = m.eval_transforms(img).unsqueeze(0).to(dev)
    with torch.inference_mode(): o = m(x)
    return o.shape[-1], "trident 自带 eval_transforms"
def gp_flash():
    m = timm.create_model("hf_hub:prov-gigapath/prov-gigapath-flash", pretrained=True, num_classes=0).eval().to(dev)
    cfg = timm.data.resolve_model_data_config(m); tf = timm.data.create_transform(**cfg, is_training=False)
    x = tf(img).unsqueeze(0).to(dev)
    with torch.inference_mode(): o = m(x)
    return o.shape[-1], f"timm in={cfg['input_size'][-1]} mean={tuple(round(v,3) for v in cfg['mean'])} crop_pct={cfg.get('crop_pct')}"
def path_found():
    try:
        import tensorflow as tf_
        note = "tensorflow %s 已在本环境" % tf_.__version__
    except Exception as e:
        return (-1, "本环境无 TensorFlow: %s" % str(e)[:60])
    from huggingface_hub import from_pretrained_keras
    m = from_pretrained_keras("google/path-foundation")
    x = np.asarray(img.resize((224, 224)), np.float32)[None] / 255.0
    o = m.signatures["serving_default"](tf_.constant(x)) if hasattr(m, "signatures") else m(x)
    v = list(o.values())[0] if isinstance(o, dict) else o
    return int(np.asarray(v).shape[-1]), note + "；输入 [0,1] 224"
print("device", dev, "| timm", timm.__version__, flush=True)
report("hibou_b", hibou_b)
report("gigapath_flash", gp_flash)
report("path_foundation", path_found)
json.dump(OK, open("/blue/qsong1/wang.qing/systema4ST/results/probe_b_ok.json", "w"), indent=1)
print("成功 %d/3" % len(OK))
