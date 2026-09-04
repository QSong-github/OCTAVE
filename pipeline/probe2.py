#!/usr/bin/env python
"""第二轮探针：对第一轮失败的几个换成作者文档给的加载方式（timm 专用配方）。"""
import torch, traceback
dev = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(2, 3, 224, 224).to(dev)
ok, bad = [], []


def try_(name, fn):
    try:
        m = fn().eval().to(dev)
        with torch.no_grad():
            f = m(x)
        if hasattr(f, "last_hidden_state"):
            f = f.last_hidden_state[:, 0]
        d = tuple(f.shape)
        print("  OK  %-14s 输出 %s" % (name, d), flush=True)
        ok.append((name, d[-1]))
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print("  --  %-14s %s" % (name, str(e).split("\n")[0][:130]), flush=True)
        bad.append(name)


import timm
print("timm", timm.__version__, "\n")
# Bioptimus 官方用法：init_values=1e-5, dynamic_img_size=False
try_("hoptimus1", lambda: timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True,
                                            init_values=1e-5, dynamic_img_size=False))
try_("hoptimus0_timm", lambda: timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True,
                                                 init_values=1e-5, dynamic_img_size=False))
try_("h0_mini", lambda: timm.create_model("hf-hub:bioptimus/H0-mini", pretrained=True,
                                          mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU))
# Paige 官方用法：SwiGLUPacked + SiLU
try_("virchow", lambda: timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True,
                                          mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU))
try_("virchow2_timm", lambda: timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True,
                                                mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU))
try_("hibou_b_timm", lambda: timm.create_model("hf-hub:histai/hibou-b", pretrained=True, num_classes=0))
print("\n本轮可用 %d 个: %s" % (len(ok), [o[0] for o in ok]))
print("仍不可用: %s" % bad)
