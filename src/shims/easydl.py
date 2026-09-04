"""easydl 的空垫片。

Hist2ST 的 transformer.py 有一行 `from easydl import *`，但该文件定义的五个类
（SelectItem / PreNorm / FeedForward / Attention / attn_block）只用到 torch 与 einops，
**没有引用 easydl 的任何符号** —— 这是作者仓库里的遗留导入。

为此装一个真包会给 hest 环境引入不必要的依赖（FINDINGS §8 工程教训①：
叠加式安装容易撞 ABI）。改用空垫片，放在独立的 shims/ 目录，
只在本驱动的 sys.path 里前置，**不修改第三方仓库、不影响其它环境**。

若将来 Hist2ST 的其它文件真的用到 easydl，导入会立即报 NameError 而非静默出错。
"""
