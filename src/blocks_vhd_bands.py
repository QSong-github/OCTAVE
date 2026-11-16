# -*- coding: utf-8 -*-
"""src/blocks_xen_bands.py run unchanged on the two Visium HD colon sections (P2, P5).
Same operator, 16x16 spatial blocks, top-200 genes, ridge and K as on Xenium; only the input and output folders differ."""
import sys
R = "/path/to/project"
sys.path.insert(0, f"{R}/src")
import blocks_xen_bands as B
B.PREP, B.EMB, B.OUTD = f"{R}/data/prepped_vhd", f"{R}/results/emb_vhd", f"{R}/results/blocks_vhd_bands"
B.main()
