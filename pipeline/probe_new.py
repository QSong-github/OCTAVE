# -*- coding: utf-8 -*-
"""扩集候选（开放权重 + 已有 token 可访问的）逐个探针：用各自官方预处理加载、前向一个批次、报输出维度。
只报成功/失败，不写任何结果文件；成功的下一步接进 hest_embed_v2.py。"""
import os, sys, json, traceback, warnings, urllib.request
warnings.filterwarnings("ignore")
import torch, timm
from PIL import Image
import numpy as np
dev = "cuda" if torch.cuda.is_available() else "cpu"
TOK = open("/path/to/.cache/huggingface/token").read().strip()
os.environ["HF_TOKEN"] = TOK; os.environ["HUGGING_FACE_HUB_TOKEN"] = TOK
img = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
OK = {}
def report(name, f):
    try:
        dim, note = f()
        OK[name] = {"dim": int(dim), "note": note}; print(f"  ✓ {name:<16s} dim={dim:<5d} {note}", flush=True)
    except Exception as e:
        print(f"  ✗ {name:<16s} {type(e).__name__}: {str(e)[:110]}", flush=True)

def timm_hub(repo, pool="cls", size=None):
    def f():
        m = timm.create_model("hf-hub:" + repo, pretrained=True, num_classes=0).eval().to(dev)
        cfg = timm.data.resolve_model_data_config(m); tf = timm.data.create_transform(**cfg, is_training=False)
        x = tf(img).unsqueeze(0).to(dev)
        with torch.inference_mode(): o = m(x)
        return o.shape[-1], f"timm in={cfg['input_size'][-1]} mean={tuple(round(v,3) for v in cfg['mean'])}"
    return f
def hf_dino(repo):
    def f():
        from transformers import AutoModel, AutoImageProcessor
        p = AutoImageProcessor.from_pretrained(repo, token=TOK); m = AutoModel.from_pretrained(repo, token=TOK).eval().to(dev)
        x = p(images=img, return_tensors="pt")["pixel_values"].to(dev)
        with torch.inference_mode(): o = m(pixel_values=x)
        return o.last_hidden_state[:, 0].shape[-1], f"hf cls mean={getattr(p,'image_mean',None)}"
    return f
def hf_clip(repo, kind="clip"):
    def f():
        from transformers import CLIPModel, CLIPImageProcessor, SiglipModel, SiglipImageProcessor
        if kind == "clip":
            p = CLIPImageProcessor.from_pretrained(repo); m = CLIPModel.from_pretrained(repo).eval().to(dev)
        else:
            p = SiglipImageProcessor.from_pretrained(repo); m = SiglipModel.from_pretrained(repo).eval().to(dev)
        x = p(images=img, return_tensors="pt")["pixel_values"].to(dev)
        with torch.inference_mode(): o = m.get_image_features(pixel_values=x)
        return o.shape[-1], f"{kind} image_features mean={getattr(p,'image_mean',None)}"
    return f
def openclip(repo):
    def f():
        import open_clip
        m, pre = open_clip.create_model_from_pretrained("hf-hub:" + repo); m = m.eval().to(dev)
        x = pre(img).unsqueeze(0).to(dev)
        with torch.inference_mode(): o = m.encode_image(x)
        return o.shape[-1], "open_clip 自带 preprocess"
    return f
def trident_enc(name):
    def f():
        from trident.patch_encoder_models.load import encoder_factory
        m = encoder_factory(name); m = m.eval().to(dev)
        x = m.eval_transforms(img).unsqueeze(0).to(dev)
        with torch.inference_mode(): o = m(x)
        return o.shape[-1], "trident 自带 eval_transforms"
    return f
def retccl():
    def f():
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("jamesdolezal/RetCCL", "retccl_torchscript.pth")
        m = torch.jit.load(p, map_location=dev).eval()
        import torchvision.transforms as T
        tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
        with torch.inference_mode(): o = m(tf(img).unsqueeze(0).to(dev))
        return (o[0] if isinstance(o,(tuple,list)) else o).shape[-1], "torchscript ResNet50 ImageNet 归一化(官方 RetCCL 用法)"
    return f
def hipt256():
    def f():
        url = "https://media.githubusercontent.com/media/mahmoodlab/HIPT/main/HIPT_4K/Checkpoints/vit256_small_dino.pth"
        dst = "/path/to/project/methods/hipt_vit256_small_dino.pth"
        if not os.path.exists(dst) or os.path.getsize(dst) < 1e6:
            os.makedirs(os.path.dirname(dst), exist_ok=True); urllib.request.urlretrieve(url, dst)
        sd = torch.load(dst, map_location="cpu")
        sd = sd.get("teacher", sd); sd = {k.replace("module.","").replace("backbone.",""): v for k,v in sd.items() if not k.startswith(("head","module.head"))}
        m = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
        miss, unexp = m.load_state_dict(sd, strict=False)
        m = m.eval().to(dev)
        import torchvision.transforms as T
        tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        with torch.inference_mode(): o = m(tf(img).unsqueeze(0).to(dev))
        return o.shape[-1], f"DINO vit_s16, 缺键{len(miss)} 多键{len(unexp)}, 文件{os.path.getsize(dst)//1e6:.0f}MB"
    return f

print("device", dev, "| timm", timm.__version__)
report("kaiko_vitb8",   timm_hub("1aurent/vit_base_patch8_224.kaiko_ai_towards_large_pathology_fms"))
report("lunit_vits16",  timm_hub("1aurent/vit_small_patch16_224.lunit_dino"))
report("lunit_r50_swav",timm_hub("1aurent/resnet50.lunit_swav"))
report("lunit_r50_bt",  timm_hub("1aurent/resnet50.lunit_bt"))
report("lunit_r50_moco",timm_hub("1aurent/resnet50.lunit_mocov2"))
report("ctranspath",    timm_hub("1aurent/swin_tiny_patch4_window7_224.CTransPath"))
report("ctranspath_tr", trident_enc("ctranspath"))
report("gpfm_tr",       trident_enc("gpfm"))
report("retccl",        retccl())
report("hipt256",       hipt256())
report("dinov2_base",   hf_dino("facebook/dinov2-base"))
report("dinov2_giant",  hf_dino("facebook/dinov2-giant"))
report("dinov3_vitb16", hf_dino("facebook/dinov3-vitb16-pretrain-lvd1689m"))
report("clip_vitl14",   hf_clip("openai/clip-vit-large-patch14"))
report("pathgen_clip",  hf_clip("jamessyx/pathgenclip-vit-large-patch14-hf"))
report("siglip2",       hf_clip("google/siglip2-so400m-patch14-384", "siglip"))
report("biomedclip",    openclip("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"))
json.dump(OK, open("/path/to/project/results/probe_new_ok.json", "w"), indent=1)
print("\n成功 %d/17 -> results/probe_new_ok.json" % len(OK))
