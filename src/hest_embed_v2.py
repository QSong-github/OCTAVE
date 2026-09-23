# -*- coding: utf-8 -*-
"""
跨 72 样本的模型半 —— HEST-benchmark 官方协议下的"报告 PCC"。

首版是我自己复刻协议, 有四处偏差, 数字全废:
  Ridge alpha 我用 1e4, 官方是 100/(D×G)≈9.8e-4 —— 差 7 个数量级, 回归被压成常数
  降维       我漏了 StandardScaler
  归一化     我用 CP10k+log1p, 官方 normalize_adata 只做 sc.pp.log1p
  对齐       我按行序, 官方按 barcode(Xenium 的 patch 数 < spot 数, 行序必错)
本版直接调用 HEST 自带的 load_adata / train_test_reg, 不再复刻。

官方默认(BenchmarkConfig): normalize=True, dimreduce='PCA', latent_dim=256, method='ridge'
划分 CSV: sample_id,patches_path,expr_path
编码器: torchvision ResNet50(ImageNet) —— HEST 默认配置内, 权重无需申请。
        编码器不是本文的变量, 之后按同一流程追加更强的即可。
"""
import os, sys, glob, json, argparse, numpy as np

print("=== 冒烟测试 ===", flush=True)
import torch, h5py
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"device {torch.cuda.get_device_name(0)}", flush=True)
    _ = (torch.rand(64, 64, device="cuda") @ torch.rand(64, 64, device="cuda")).sum().item()

sys.path.insert(0, "/path/to/he2st/HEST/src")
from hest.bench.st_dataset import load_adata            # 官方: barcode 子集 + log1p
from hest.bench.trainer import train_test_reg           # 官方: alpha=100/(D*G), lsqr, no intercept
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

B = "/path/to/he2st/HEST/eval/bench_data"
EMB = "/path/to/project/results/hest_emb"


# 全部开放权重(无需 HF 授权)。UNI/Virchow2/GigaPath/H-optimus/CONCH 是 gated,
# 本地也无缓存 —— HEST 这条线暂时覆盖不到, 需账号授权。
# Visium HD 那条线用上游项目已提取的 .npy 嵌入, 不受此限(见 tower_sweep.py 的 25 塔)。
HF_REPOS = {"phikon": "owkin/phikon", "phikon_v2": "owkin/phikon-v2",
            "hibou_b": "histai/hibou-b",
            "dinov2_large": "facebook/dinov2-large",
            "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "phaet": "wearewaiv/phaet", "mascaret": "wearewaiv/mascaret",
            "dinov2_base": "facebook/dinov2-base", "dinov2_giant": "facebook/dinov2-giant",
            "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m", "dinov3_vith16": "facebook/dinov3-vith16plus-pretrain-lvd1689m"}
TIMM_SIZE = {"kaiko_vitl14": 518, "musk": 384}      # 该模型 pretrain 于 518, 224 会断言失败
TIMM_REPOS = {"lunit_vits8": "hf_hub:1aurent/vit_small_patch8_224.lunit_dino",
              "kaiko_vitb16": "hf_hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms",
              "kaiko_vits16": "hf_hub:1aurent/vit_small_patch16_224.kaiko_ai_towards_large_pathology_fms",
              "kaiko_vitl14": "hf_hub:1aurent/vit_large_patch14_reg4_dinov2.kaiko_ai_towards_large_pathology_fms"}
CIGA_CKPT = "/path/to/project/stflow_run/weights/fm_v1/ciga/tenpercent_resnet18.ckpt"
# trident 路径：uni/virchow/gigapath/conch 等 gated 权重已在本地 HF 缓存，
# HF_HUB_OFFLINE=1 即可加载（实测 13/14 可用，仅 gpfm 的 checkpoint 损坏）。
TRIDENT_ENC = ["uni_v1", "uni_v2", "virchow", "virchow2", "gigapath", "conch_v1",
               "conch_v15", "hoptimus0", "keep", "midnight12k", "openmidnight", "hibou_l",
               "ctranspath", "gpfm"]
# 2026-08-29 新增，加载与预处理均取自各模型官方文档，探针已逐个验证前向。
# 关键：不套 ImageNet mean/std。Bioptimus 系列与 CLIP 系列都有自己的统计量，
# 套错会静默降低该模型的表现（与上面 trident 分支的注释同理）。
NEW_TIMM = {"hoptimus1": ("hf-hub:bioptimus/H-optimus-1", "cls"),
            "h0_mini":   ("hf-hub:bioptimus/H0-mini", "cls_mean")}
NEW_CLIP = {"plip": "vinid/plip", "quiltnet": "wisdomik/QuiltNet-B-32",
            "clip_vitl14": "openai/clip-vit-large-patch14", "pathgen_clip": "jamessyx/pathgenclip-vit-large-patch14-hf"}
NEW_HF2 = {"genbio_pathfm": "genbio-ai/genbio-pathfm"}
# OmiCLIP（Nature Methods 2025）：220 万对 Visium 图块与表达训练，直接针对
# H&E 与空间转录组的对齐，是本文任务域内最贴题的编码器。权重开放。
OMICLIP_CKPT = "/path/to/project/methods/OmiCLIP/checkpoint.pt"
# 2026-09-03 扩集：timm hf-hub（各自 data config 预处理）、SigLIP2、BiomedCLIP、RetCCL、更多 HF/CLIP 通用基线
NEW_TIMM2 = {"kaiko_vitb8": "hf-hub:1aurent/vit_base_patch8_224.kaiko_ai_towards_large_pathology_fms",
             "lunit_vits16": "hf-hub:1aurent/vit_small_patch16_224.lunit_dino",
             "lunit_r50_swav": "hf-hub:1aurent/resnet50.lunit_swav",
             "lunit_r50_bt": "hf-hub:1aurent/resnet50.lunit_bt",
             "lunit_r50_moco": "hf-hub:1aurent/resnet50.lunit_mocov2",
             # 2026-09-04：DistillPath（arXiv 2608.17872），Virchow2 蒸馏到 ViT-S/16；KS16 以 kaiko 初始化（mean/std 0.5），IS16 以 IN21k 初始化（ImageNet 统计量）
             "distillpath_ks16": "hf_hub:RamonK/DistillPath-KS16-Virchow2",
             "distillpath_is16": "hf_hub:RamonK/DistillPath-IS16-Virchow2",
             # 2026-09-04：Pathryoshka-B（arXiv 2511.23204），三教师套娃蒸馏 ViT-B/14+registers；模型卡用 timm 的 data config（ImageNet、224）
             "pathryoshka_b": "hf-hub:SchuefflerLab/Pathryoshka-B"}
NEW_SIGLIP = {"siglip2": "google/siglip2-so400m-patch14-384"}
NEW_OPENCLIP = {"biomedclip": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"}
RETCCL = {"retccl": ("jamesdolezal/RetCCL", "retccl_torchscript.pth")}
NEW_ENC = (list(NEW_TIMM) + list(NEW_CLIP) + list(NEW_HF2) + ["omiclip", "omiclip_raw", "musk"]
           + list(NEW_TIMM2) + list(NEW_SIGLIP) + list(NEW_OPENCLIP) + list(RETCCL))
# 2026-09-03 第二批。hibou_b 的 config 没有 pixel_mean，但 preprocessor_config 里的 image_mean/std
# 是 (0.7068,0.5755,0.722)/(0.195,0.2316,0.1816)，不是 ImageNet 的 —— 走 HF_REPOS 的兜底会套错统计量，
# 故单独一支：AutoModel + AutoImageProcessor 的统计量。
HF_PROC = {"hibou_b": "histai/hibou-b"}
# gigapath_flash：权重是 timm 风格 (config.json + pytorch_model.bin)，但架构名 gigapath_tile_enc_dinov2s
# 只在作者的 gigapath 包里注册（要求 timm>=1.0.3）。这里按作者 gigapath/tile_encoder.py 的 _TILE_ENC_ARGS
# 直接构造 timm VisionTransformer 并严格加载 state_dict（缺/多任何键都报错）。预处理同 gigapath：ImageNet、224。
GIGA_FLASH = {"gigapath_flash": "prov-gigapath/prov-gigapath-flash"}
# mSTAR（Nat Commun 2025，ViT-L/16，0.3B）：模型卡的加载方式与预处理（ImageNet 统计量、224）。
# 其 config.json 的 pretrained_cfg.mean/std 是 0.5/0.5（augreg 标签遗留），模型卡的 transform 写的是 ImageNet，以模型卡为准，
# 所以不走 NEW_TIMM2 的 resolve_model_data_config。
MSTAR = {"mstar": "hf-hub:Wangyh/mSTAR"}
# LitePath / LiteFM（2026-02，arXiv 2602.14010）：Virchow2+H-optimus-1+UNI2 三教师蒸馏；四个发布的图块编码器。
LITEFM_DIR = "/path/to/project/methods/LitePath/inference"
LITEFM = {"litefm": ("small", "LiteFM.pth"), "litefm_s": ("tiny", "LiteFM-S.pth"),
          "litefm_l": ("base", "LiteFM-L.pth"), "litevirchow2": ("small", "LiteVirchow2.pth")}
ALL_ENC = ["resnet50", "ciga"] + list(HF_REPOS) + list(TIMM_REPOS) + TRIDENT_ENC + NEW_ENC + list(GIGA_FLASH) + list(MSTAR) + list(LITEFM)


def encoder(name, dev):
    if name == "resnet50":
        import torchvision
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = torch.nn.Identity()
        return m.eval().to(dev), None
    if name == "ciga":            # Ciga 等人自监督病理 ResNet18(GitHub release, 开放)
        import torchvision, torch.nn as nn
        m = torchvision.models.resnet18(weights=None); m.fc = nn.Identity()
        sd = torch.load(CIGA_CKPT, map_location="cpu")["state_dict"]
        sd = {k.replace("model.resnet.", ""): v for k, v in sd.items() if "fc." not in k}
        m.load_state_dict(sd, strict=True)
        return m.eval().to(dev), None
    if name == "musk":
        import sys as _sys, torchvision as _tv
        _sys.path.insert(0, "/path/to/methods/MUSK")
        from musk import utils as _mu, modeling as _mm      # noqa: F401  注册 timm 模型
        from timm.models import create_model as _cm
        from timm.data.constants import IMAGENET_INCEPTION_MEAN as _M, IMAGENET_INCEPTION_STD as _S
        mk = _cm("musk_large_patch16_384")
        _mu.load_model_and_may_interpolate("hf_hub:xiangjx/musk", mk, "model|module", "")
        mk.to(device=dev, dtype=torch.float16).eval()
        tf = _tv.transforms.Compose([
            _tv.transforms.Resize(384, interpolation=3, antialias=True),
            _tv.transforms.CenterCrop((384, 384)),
            _tv.transforms.ToTensor(),
            _tv.transforms.Normalize(mean=_M, std=_S)])
        return mk, ("musk", tf, "musk")
    if name in ("omiclip", "omiclip_raw"):
        # 该 checkpoint 存了 numpy scalar；PyTorch 2.6 起 torch.load 默认 weights_only=True
        # 会拒绝。open_clip 3.3 的 create_model_from_pretrained 有 weights_only 参数，
        # 直接传 False（权重来自作者官方 HF 仓库）。
        from open_clip import create_model_from_pretrained
        m, pre = create_model_from_pretrained("coca_ViT-L-14", device=dev,
                                              pretrained=OMICLIP_CKPT,
                                              weights_only=False)
        return m.eval().to(dev), ("omiclip", pre, name)
    if name in NEW_TIMM:
        import timm
        from timm.data import resolve_model_data_config, create_transform
        repo, pool = NEW_TIMM[name]
        kw = dict(pretrained=True)
        if "H-optimus" in repo:
            kw.update(init_values=1e-5, dynamic_img_size=False)
        else:
            kw.update(mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)
        m = timm.create_model(repo, **kw).eval().to(dev)
        tf = create_transform(**resolve_model_data_config(m), is_training=False)
        return m, ("own", tf, pool)
    if name in NEW_TIMM2:
        # 普通 timm hf-hub：num_classes=0 得到池化特征；预处理取模型自带 data config（Kaiko/Lunit 的 mean/std 各不相同）
        import timm
        from timm.data import resolve_model_data_config, create_transform
        m = timm.create_model(NEW_TIMM2[name], pretrained=True, num_classes=0).eval().to(dev)
        tf = create_transform(**resolve_model_data_config(m), is_training=False)
        return m, ("own", tf, "auto")
    if name in NEW_SIGLIP:
        # SiglipModel 的 get_image_features / SiglipImageProcessor 接口与 CLIP 相同，复用 clip 前向
        from transformers import SiglipModel, SiglipImageProcessor
        m = SiglipModel.from_pretrained(NEW_SIGLIP[name]).eval().to(dev)
        pr = SiglipImageProcessor.from_pretrained(NEW_SIGLIP[name])
        return m, ("clip", pr, "clip")
    if name in NEW_OPENCLIP:
        # open_clip hf-hub，自带 preprocess；encode_image 走 omiclip 前向，normalize=False（同 omiclip_raw 口径）
        from open_clip import create_model_from_pretrained
        m, pre = create_model_from_pretrained(NEW_OPENCLIP[name], device=dev)
        return m.eval().to(dev), ("omiclip", pre, name)
    if name in RETCCL:
        # 作者移植的 torchscript ResNet50；官方 RetCCL 用 ImageNet 归一化、224 输入
        from huggingface_hub import hf_hub_download
        import torch.nn as nn, torchvision.transforms as _T
        repo, fn = RETCCL[name]
        ts = torch.jit.load(hf_hub_download(repo, fn), map_location=dev).eval()
        class _W(nn.Module):
            def __init__(self, ts): super().__init__(); self.ts = ts
            def forward(self, x):
                o = self.ts(x); return o[0] if isinstance(o, (tuple, list)) else o
        tf = _T.Compose([_T.Resize(224), _T.CenterCrop(224), _T.ToTensor(),
                         _T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        return _W(ts).eval().to(dev), ("own", tf, "auto")
    if name in NEW_CLIP:
        from transformers import CLIPModel, CLIPImageProcessor
        m = CLIPModel.from_pretrained(NEW_CLIP[name]).eval().to(dev)
        pr = CLIPImageProcessor.from_pretrained(NEW_CLIP[name])
        return m, ("clip", pr, "clip")
    if name in NEW_HF2:
        from transformers import AutoModel
        from torchvision import transforms as _T
        m = AutoModel.from_pretrained(NEW_HF2[name], trust_remote_code=True).eval().to(dev)
        # genbio-pathfm 的 model card 明确给出自己的统计量，不是 ImageNet 的。
        tf = _T.Compose([_T.Resize((224, 224)), _T.ToTensor(),
                         _T.Normalize(mean=(0.697, 0.575, 0.728), std=(0.188, 0.240, 0.187))])
        return m, ("own", tf, "auto")
    if name in HF_PROC:
        from transformers import AutoModel, AutoImageProcessor
        m = AutoModel.from_pretrained(HF_PROC[name], trust_remote_code=True).eval().to(dev)
        pr = AutoImageProcessor.from_pretrained(HF_PROC[name], trust_remote_code=True)
        pm, ps = list(pr.image_mean), list(pr.image_std)
        print("  %s 自带统计量 mean=%s std=%s" % (name, pm, ps), flush=True)
        v = lambda t: torch.tensor(t, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
        return m, ("hfcfg", v(pm), v(ps))
    if name in MSTAR:
        import timm, torchvision.transforms as _T
        m = timm.create_model(MSTAR[name], pretrained=True, init_values=1e-5, dynamic_img_size=True,
                              num_classes=0).eval().to(dev)
        tf = _T.Compose([_T.Resize(224), _T.ToTensor(),
                         _T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        return m, ("own", tf, "auto")
    if name in LITEFM:
        import importlib.util, os as _os
        spec = importlib.util.spec_from_file_location("litefm_authors", _os.path.join(LITEFM_DIR, "models", "litefm.py"))
        L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
        size, ck = LITEFM[name]
        ctor = {"tiny": L.custom_vit_tiny_patch16_224, "small": L.custom_vit_small_patch16_224, "base": L.custom_vit_base_patch16_224}[size]
        m = ctor(dev, _os.path.join(LITEFM_DIR, "ckpts", ck), proj_dim=1024, out_dim_dict=None).eval()
        return m, ("own", L.get_litefm_trans(), "auto")
    if name in GIGA_FLASH:
        import torch.nn as nn, torchvision.transforms as _T
        from timm.layers import SwiGLUPacked
        from timm.models.vision_transformer import VisionTransformer
        from huggingface_hub import hf_hub_download
        m = VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6,
                              mlp_ratio=2048 / 384.0, mlp_layer=SwiGLUPacked, act_layer=nn.SiLU,
                              init_values=1e-5, num_classes=0, global_pool="token",
                              class_token=True, reg_tokens=0)
        sd = torch.load(hf_hub_download(GIGA_FLASH[name], "pytorch_model.bin"), map_location="cpu")
        if isinstance(sd, dict) and "patch_embed.proj.weight" not in sd and "model" in sd:
            sd = sd["model"]
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, ("gigapath_flash 权重键不对应", missing[:5], unexpected[:5])
        print("  gigapath_flash: %d 个张量严格对上" % len(sd), flush=True)
        tf = _T.Compose([_T.Resize(224), _T.CenterCrop(224), _T.ToTensor(),
                         _T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        return m.eval().to(dev), ("own", tf, "auto")
    if name in TRIDENT_ENC:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from trident.patch_encoder_models.load import encoder_factory
        e = encoder_factory(name).to(dev).eval()
        return e, ("trident", e.eval_transforms, getattr(e, "precision", torch.float32))  # kind 为元组
    if name in TIMM_REPOS:
        import timm
        m = timm.create_model(TIMM_REPOS[name], pretrained=True, num_classes=0)
        return m.eval().to(dev), None
    from transformers import AutoModel
    m = AutoModel.from_pretrained(HF_REPOS[name], trust_remote_code=True).eval().to(dev)
    pm, ps = getattr(m.config, "pixel_mean", None), getattr(m.config, "pixel_std", None)
    if pm is None or ps is None:
        return m, "hf"
    print("  %s 自带统计量 mean=%s std=%s" % (name, pm, ps), flush=True)
    v = lambda t: torch.tensor(t, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
    return m, ("hfcfg", v(pm), v(ps))


@torch.no_grad()
def embed_and_barcodes(model, kind, path, dev, bs=256, size=224):
    with h5py.File(path, "r") as h:
        bk = "barcodes" if "barcodes" in h else "barcode"
        bc = np.asarray(h[bk][:]).flatten().astype(str).tolist()
        n = h["img"].shape[0]
        mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
        out = []
        for i in range(0, n, bs):
            x = torch.tensor(np.asarray(h["img"][i:i + bs]), device=dev).float()
            x = x[..., :3].permute(0, 3, 1, 2) / 255.0
            if size != 224:
                x = torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)
            if isinstance(kind, tuple) and kind[0] == "musk":
                from PIL import Image
                arr = np.asarray(h["img"][i:i + bs])[..., :3].astype(np.uint8)
                xb = torch.stack([kind[1](Image.fromarray(a_)) for a_ in arr]).to(dev, dtype=torch.float16)
                with torch.inference_mode():
                    f = model(image=xb, with_head=False, out_norm=False,
                              ms_aug=True, return_global=True)[0]
                f = f.float()
            elif isinstance(kind, tuple) and kind[0] == "omiclip":
                from PIL import Image
                import torch.nn.functional as _F
                arr = np.asarray(h["img"][i:i + bs])[..., :3].astype(np.uint8)
                xb = torch.stack([kind[1](Image.fromarray(a_)) for a_ in arr]).to(dev)
                # CoCa 的 encode_image 默认 normalize=True。作者在 encode_images 里
                # 也做 L2，那是为检索（余弦相似度）。本文用途是岭回归，因此另跑一个
                # normalize=False 的变体，与其余编码器同口径，取对该模型有利的报。
                f = model.encode_image(xb, normalize=(kind[2] == "omiclip")).float()
            elif isinstance(kind, tuple) and kind[0] in ("own", "clip"):
                from PIL import Image
                arr = np.asarray(h["img"][i:i + bs])[..., :3].astype(np.uint8)
                tag, pre, pool = kind
                if tag == "clip":
                    xb = pre(images=[Image.fromarray(a_) for a_ in arr],
                             return_tensors="pt")["pixel_values"].to(dev)
                    f = model.get_image_features(pixel_values=xb)
                else:
                    if pre is None:                       # 该模型无 timm 配置，退回 ImageNet 统计量
                        xb = (x - mean) / std
                    else:
                        xb = torch.stack([pre(Image.fromarray(a_)) for a_ in arr]).to(dev)
                    o = model(xb.contiguous())   # 该模型内部对入参做 .view()，要求连续
                    if pool == "cls_mean":                # 作者配方：[CLS] 与 patch 均值拼接
                        f = torch.cat([o[:, 0], o[:, 1:].mean(1)], dim=-1)
                    elif hasattr(o, "last_hidden_state"):
                        f = o.last_hidden_state[:, 0]
                    elif hasattr(o, "pooler_output"):
                        f = o.pooler_output
                    else:
                        f = o
                f = f.float()
            elif isinstance(kind, tuple) and kind[0] == "trident":
                # 每个 trident 塔有自己的 eval_transforms（缩放 + 归一化各不相同）。
                # 套用 ImageNet mean/std 会静默降低该塔的性能 —— 必须用它自己的。
                from PIL import Image
                tf, prec = kind[1], kind[2]
                arr = np.asarray(h["img"][i:i + bs])[..., :3].astype(np.uint8)
                xb = torch.stack([tf(Image.fromarray(a)) for a in arr]).to(dev)
                use_ac = prec in (torch.float16, torch.bfloat16)
                with torch.autocast("cuda", dtype=prec, enabled=use_ac and dev == "cuda"):
                    f = model(xb)
                f = f.float()
            elif isinstance(kind, tuple) and kind[0] == "hfcfg":
                o = model(pixel_values=(x - kind[1]) / kind[2])
                f = o.last_hidden_state[:, 0] if hasattr(o, "last_hidden_state") else o.pooler_output
            elif kind == "hf":
                o = model(pixel_values=(x - mean) / std)
                f = o.last_hidden_state[:, 0] if hasattr(o, "last_hidden_state") else o.pooler_output
            else:
                f = model((x - mean) / std)
            out.append(f.cpu().numpy())
    return np.concatenate(out).astype(np.float32), bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="phikon", choices=ALL_ENC)
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=256,
                    help="kaiko_vitl14 在 518×518 下 256 会 CUDA OOM，用 64")
    args = ap.parse_args()
    args.out = args.out or f"results/hest_reported_pcc_{args.encoder}.json"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, kind = encoder(args.encoder, dev)
    os.makedirs(EMB, exist_ok=True)

    per_sample, cohort_of = {}, {}
    for c in sorted(os.listdir(B)):
        if not os.path.isdir(os.path.join(B, c, "adata")):
            continue
        genes = json.load(open(os.path.join(B, c, "var_50genes.json")))["genes"]
        cache = {}

        def get(sid):
            if sid in cache:
                return cache[sid]
            f = os.path.join(EMB, f"{sid}_{args.encoder}.npz")
            if os.path.exists(f):
                z = np.load(f, allow_pickle=True); X, bc = z["X"], z["bc"].tolist()
            else:
                X, bc = embed_and_barcodes(model, kind, os.path.join(B, c, "patches", f"{sid}.h5"),
                                           dev, bs=args.batch, size=TIMM_SIZE.get(args.encoder, 224))
                np.savez(f, X=X, bc=np.array(bc, dtype=object))
            # 官方: 按 barcode 子集 adata, 只取 50 基因, log1p
            Y = load_adata(os.path.join(B, c, "adata", f"{sid}.h5ad"),
                           genes=genes, barcodes=bc, normalize=True).values.astype(np.float32)
            assert X.shape[0] == Y.shape[0], f"{sid}: emb {X.shape[0]} vs expr {Y.shape[0]}"
            cache[sid] = (X, Y)
            cohort_of[sid] = c
            print(f"  {c}/{sid}: X{X.shape} Y{Y.shape}", flush=True)
            return cache[sid]

        n_splits = len(glob.glob(os.path.join(B, c, "splits", "test_*.csv")))
        for k in range(n_splits):
            rd = lambda f: [l.split(",")[0] for l in
                            open(os.path.join(B, c, "splits", f)).read().splitlines()[1:] if l.strip()]
            tr, te = rd(f"train_{k}.csv"), rd(f"test_{k}.csv")
            Xtr = np.concatenate([get(s)[0] for s in tr]); Ytr = np.concatenate([get(s)[1] for s in tr])
            pipe = Pipeline([("scaler", StandardScaler()),
                             ("PCA", PCA(n_components=min(args.latent_dim, Xtr.shape[1],
                                                          Xtr.shape[0] - 1), random_state=0))]).fit(Xtr)
            Ztr = pipe.transform(Xtr)
            for s in te:
                Xs, Ys = get(s)
                r, _ = train_test_reg(Ztr, pipe.transform(Xs), Ytr, Ys, genes=genes, method="ridge")
                v = float(r["pearson_mean"])
                per_sample.setdefault(s, []).append(v)
                print(f"  [{c} fold{k}] test={s} pearson_mean={v:.4f}", flush=True)

    out = {s: {"cohort": cohort_of[s], "folds": v, "pcc": float(np.mean(v))}
           for s, v in per_sample.items()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n=== {len(out)} 样本 ===")
    for c in sorted(set(cohort_of.values())):
        v = [d["pcc"] for d in out.values() if d["cohort"] == c]
        print(f"  {c:10s} n={len(v):3d} PCC 均值={np.mean(v):.4f} 范围=[{min(v):.4f},{max(v):.4f}]")
    print(f"已存 {args.out}")


if __name__ == "__main__":
    main()
