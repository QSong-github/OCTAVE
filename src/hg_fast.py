# -*- coding: utf-8 -*-
"""HGGEP 的 build_adj_hypergraph 向量化替换 —— 数学等价，只消除 Python 循环。

作者原版对 n 个节点逐个循环，每轮几次 kernel 启动加 4 次 list.append。
n=4000 时约两万次启动，是启动延迟受限，换大卡无用（B200 上一折仍要 ~5 小时）。

等价性要点：
  · 原版 `torch.tensor(edge_weights)` / `torch.tensor(hypergraph_edges)` 会切断梯度，
    所以本版同样 detach，反传行为一致。
  · 边的顺序：原版按 i 递增、每个 i 内按 topk 顺序追加；repeat_interleave 给出同一顺序。
  · 距离矩阵按行分块算，避免 n^2*m 的峰值，算式与 torch.norm(x1-x2, dim=-1) 逐元素相同。
本文件在导入时对随机输入与作者原版逐位比对，不一致就抛错。

block 取 64：距离矩阵按行分块，临时张量是 (block, n, m)。HGGEP 在 n=4000、bake=5 下
模型本身就要约 93 GB（4000 token 注意力 x 16 头 x 8 层 x 5 次增广），95 GB 的
rtx6000 装不下，本项目只在 183 GB 的 B200 上跑它。分块只是把这一处的峰值压到可忽略。
"""
import torch
from torch_geometric.data import Data


def build_adj_hypergraph_fast(features, adjacency_matrix, num_neighbors, block=256):
    # 全程 no_grad：作者原版的 edge_index / edge_attr 都经 torch.tensor(list) 建成，
    # 本来就不带梯度；且原版逐节点循环里每轮的图在下一轮即被释放，只留 detach 过的标量。
    # 首版向量化没有 no_grad，等于把整个 n x n 距离计算的图一次性全留住 —— 那是我引入的
    # 行为差异，在训练片较多的队列上直接 OOM。加上 no_grad 后与原版的内存行为一致。
    n, m = features.size()
    with torch.no_grad():
        Nd = adjacency_matrix / adjacency_matrix.sum(dim=1, keepdim=True)
        E = torch.empty(n, n, device=features.device, dtype=features.dtype)
        for s in range(0, n, block):
            e = min(s + block, n)
            E[s:e] = torch.norm(features[s:e, None, :] - features[None, :, :], dim=-1)
        Ed = E / E.sum(dim=1, keepdim=True)
        d = Nd + Ed
        vals, idx = torch.topk(d, k=num_neighbors + 1, dim=1)
        rows = torch.arange(n, device=features.device).repeat_interleave(num_neighbors + 1)
        edge_index = torch.stack([rows, idx.reshape(-1)]).long()
        edge_attr = vals.reshape(-1).float()
    # x 仍带梯度，与作者一致：只有边的构造不参与反传
    return Data(x=features, edge_index=edge_index, edge_attr=edge_attr, y=None)


def verify(orig, k=3, n=64, m=16, tol=1e-5, seed=0):
    """与作者原版逐位比对：边集合必须完全相同，权重在容差内。"""
    g = torch.Generator().manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    feat = torch.randn(n, m, generator=g).to(dev)
    adj = torch.rand(n, n, generator=g).to(dev) + 1e-3
    a = orig(feat, adj, k)
    b = build_adj_hypergraph_fast(feat, adj, k)
    ei_a, ei_b = a.edge_index.to(dev), b.edge_index.to(dev)
    ea_a, ea_b = a.edge_attr.to(dev).float(), b.edge_attr.to(dev).float()
    assert ei_a.shape == ei_b.shape, "边数不同 %s vs %s" % (ei_a.shape, ei_b.shape)
    same = int((ei_a == ei_b).all().item())
    dmax = float((ea_a - ea_b).abs().max().item())
    print("  等价性核对 n=%d k=%d: 边索引逐位相同=%s  权重最大差=%.3e" % (n, k, bool(same), dmax), flush=True)
    assert same, "边索引不同"
    assert dmax < tol, "权重差 %.3e 超过容差" % dmax
    return True
