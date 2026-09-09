# -*- coding: utf-8 -*-
"""Xenium 区域的通用塔嵌入：复用 src/hest_embed_v2.encoder() 的全部加载分支（trident / timm / HF / CLIP / SigLIP / MUSK / OmiCLIP / 自定义），
逐 kind 的前向逻辑与 hest_embed_v2.embed_and_barcodes 逐行同构，只把输入从 h5 的 224 块换成 openslide 在 level 0 按 obsm['pxl'] 切出的 ctx_px 方块
（物理视野 61.4 µm，与 hd_embed.py 相同）。输出 results/emb_xen/emb_{enc}_{region}.npy。"""
import os, sys, argparse, numpy as np, torch, anndata as ad
sys.path.insert(0, "/blue/qsong1/wang.qing/systema4ST/src")
import hest_embed_v2 as HE
def embed_batch(model, kind, arr, dev, size=224):
    from PIL import Image
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    x = torch.tensor(arr, device=dev).float()[..., :3].permute(0, 3, 1, 2) / 255.0
    if x.shape[-1] != size: x = torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)
    if isinstance(kind, tuple) and kind[0] == "musk":
        xb = torch.stack([kind[1](Image.fromarray(a_)) for a_ in arr]).to(dev, dtype=torch.float16)
        with torch.inference_mode(): f = model(image=xb, with_head=False, out_norm=False, ms_aug=True, return_global=True)[0]
        return f.float()
    if isinstance(kind, tuple) and kind[0] == "omiclip":
        xb = torch.stack([kind[1](Image.fromarray(a_)) for a_ in arr]).to(dev); return model.encode_image(xb, normalize=(kind[2] == "omiclip")).float()
    if isinstance(kind, tuple) and kind[0] in ("own", "clip"):
        tag, pre, pool = kind
        if tag == "clip":
            xb = pre(images=[Image.fromarray(a_) for a_ in arr], return_tensors="pt")["pixel_values"].to(dev); return model.get_image_features(pixel_values=xb).float()
        xb = (x - mean) / std if pre is None else torch.stack([pre(Image.fromarray(a_)) for a_ in arr]).to(dev)
        o = model(xb.contiguous())
        if pool == "cls_mean": f = torch.cat([o[:, 0], o[:, 1:].mean(1)], dim=-1)
        elif hasattr(o, "last_hidden_state"): f = o.last_hidden_state[:, 0]
        elif hasattr(o, "pooler_output"): f = o.pooler_output
        else: f = o
        return f.float()
    if isinstance(kind, tuple) and kind[0] == "trident":
        tf, prec = kind[1], kind[2]; xb = torch.stack([tf(Image.fromarray(a_)) for a_ in arr]).to(dev); use_ac = prec in (torch.float16, torch.bfloat16)
        with torch.autocast("cuda", dtype=prec, enabled=use_ac and dev == "cuda"): f = model(xb)
        return f.float()
    if isinstance(kind, tuple) and kind[0] == "hfcfg":
        o = model(pixel_values=(x - kind[1]) / kind[2]); return (o.last_hidden_state[:, 0] if hasattr(o, "last_hidden_state") else o.pooler_output).float()
    if kind == "hf":
        o = model(pixel_values=(x - mean) / std); return (o.last_hidden_state[:, 0] if hasattr(o, "last_hidden_state") else o.pooler_output).float()
    return model((x - mean) / std).float()
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--h5ad", required=True); ap.add_argument("--tiff", required=True); ap.add_argument("--out", required=True); ap.add_argument("--encoder", required=True); ap.add_argument("--ctx_px", type=int, default=224); ap.add_argument("--batch", type=int, default=128); a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import openslide
    A = ad.read_h5ad(a.h5ad, backed="r"); pxl = np.asarray(A.obsm["pxl"], np.float64); print(f"[{os.path.basename(a.h5ad)}] {len(pxl)} bin enc={a.encoder} ctx={a.ctx_px} {dev}", flush=True)
    model, kind = HE.encoder(a.encoder, dev); sl = openslide.OpenSlide(a.tiff); half = a.ctx_px // 2; out = []
    SIZE = getattr(HE, "TIMM_SIZE", {}).get(a.encoder, 224); print(f"input size {SIZE}", flush=True)
    with torch.no_grad():
        for i in range(0, len(pxl), a.batch):
            pb = pxl[i:i + a.batch]
            arr = np.stack([np.asarray(sl.read_region((int(x - half), int(y - half)), 0, (a.ctx_px, a.ctx_px)).convert("RGB"), dtype=np.uint8) for x, y in pb])
            out.append(embed_batch(model, kind, arr, dev, size=SIZE).cpu().numpy().astype(np.float32))
            if (i // a.batch) % 100 == 0: print(f"    {i + len(pb)}/{len(pxl)}", flush=True)
    E = np.concatenate(out); os.makedirs(os.path.dirname(a.out), exist_ok=True); np.save(a.out, E); print(f"已存 {a.out} {E.shape}", flush=True)
