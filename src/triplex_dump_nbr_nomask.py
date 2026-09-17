# -*- coding: utf-8 -*-
"""个别切片的邻居视图在组织掩膜下几乎全被丢弃（TENX111 保留 0 片、NCBI684 保留 833/3487），
原因是 1120 px（约 560 µm）的窗口大量越出掩膜轮廓。掩膜只是切块时的取舍，不是模型的一部分，
故对这些片改用 use_mask=False 重切邻居块，其余与作者 save_patches 完全一致（224×5 px、0.5 µm/px、随后 match_to_target）。"""
import sys, os
sys.path.insert(0, "/path/to/systema4ST/methods/hest_new")
sys.path.insert(0, "/path/to/systema4ST/methods/triplex_shim")
sys.path.insert(0, "/path/to/systema4ST/methods/TRIPLEX/src")
from hest import iter_hest
from preprocess.prepare_data import match_to_target
C, sid = sys.argv[1], sys.argv[2]
H = "/path/to/systema4ST/data/hest_wsis"; out = f"/path/to/systema4ST/data/triplex/{C}"
st = [s for s in iter_hest(H, id_list=[sid])][0]
if st._tissue_contours is None: st.segment_tissue(method="deep")
p = f"{out}/patches/neighbor/{sid}.h5"
if os.path.exists(p): os.remove(p)
# HEST 的 dump_patches 在 use_mask=False 时 valid_barcodes 未赋值（只在掩膜分支里设），这里照其 use_mask=True 的路径原样重写，仅把 mask 置为 None。
import numpy as np
adata = st.adata.copy(); src = st.pixel_size; tps, dps = 224 * 5, 0.5
psrc = tps * (dps / src)
ctr = adata.obsm["spatial"]; tl = ctr - psrc // 2
ins = (0 <= tl[:, 0] + psrc) & (tl[:, 0] < st.wsi.width) & (0 <= tl[:, 1] + psrc) & (tl[:, 1] < st.wsi.height)
tl = np.array(tl[ins]).astype(int); barcodes = np.array(adata.obs.index)[ins]
patcher = st.wsi.create_patcher(tps, src, dps, mask=None, custom_coords=tl, threshold=0)
patcher.to_h5(p, extra_assets={"barcode": barcodes})
match_to_target(f"{out}/patches/{sid}.h5", p)
import h5py, numpy as np
with h5py.File(p) as h: print(f"{C}/{sid}: 邻居块 {len(np.asarray(h['barcode'][:]).flatten())} 片（无掩膜）", flush=True)
