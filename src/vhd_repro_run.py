# -*- coding: utf-8 -*-
"""rerun the old Visium HD analyses of the paper on the rebuilt inputs in data/vhd_legacy .
The old scripts read the parent project's binned_16um.h5ad and features through st_encoder_bench (H5AD, EMBDIR), which src/effres.py
points at an upstream folder that no longer exists. Here both are pointed at data/vhd_legacy before the script runs, and every output goes under
results/repro_vhd, so no stored result is overwritten.
  ruler32         src/ruler_fold.py (Hibou-L, tmax 2048)                       -> results/repro_vhd/t32/ruler_fold.json
  fig1            fig1_windows.py on that ruler_fold.json, H&E read from the pyramidal TIFF -> results/repro_vhd/fig1/
  sweep [args]    src/tower_sweep.py with the given arguments
  buffer          src/buffer_cv.py                                             -> results/repro_vhd/buffer_cv.json
Before the task name's arguments, --emb <folder> picks the feature folder under data/vhd_legacy (default emb)."""
import os, sys
R = "/path/to/project"
sys.path.insert(0, f"{R}/src")
import st_encoder_bench as SEB, effres   # effres points SEB.H5AD and SEB.EMBDIR at the parent project; overridden below
OUT = f"{R}/results/repro_vhd"
task, rest = sys.argv[1], sys.argv[2:]
emb = "emb"
if rest[:1] == ["--emb"]:
    emb, rest = rest[1], rest[2:]
SEB.H5AD, SEB.EMBDIR = f"{R}/data/vhd_legacy/binned_16um.h5ad", f"{R}/data/vhd_legacy/{emb}"
print(f"task {task}: H5AD {SEB.H5AD}, EMBDIR {SEB.EMBDIR}, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}", flush=True)


def run_patched(path, reps, argv, extra=None):
    """Run a script as __main__ after replacing each string in reps, which must occur exactly once."""
    src = open(path).read()
    for a, b in reps:
        assert src.count(a) == 1, f"{path}: expected one match of {a!r}, found {src.count(a)}"
        src = src.replace(a, b)
    sys.argv = argv
    g = {"__name__": "__main__", "__file__": path}
    g.update(extra or {})
    exec(compile(src, path, "exec"), g)


if task == "ruler32":
    import ruler_fold
    sys.argv = ["ruler_fold.py", "--tower", "hibou_l", "--tmax", "2048", "--out", f"{OUT}/t32"]
    ruler_fold.main()
elif task == "fig1":
    # FIG1_OUT sends a rerun to another folder, so the first rerun's files stay for comparison
    O = os.environ.get("FIG1_OUT", f"{OUT}/fig1")
    os.makedirs(O, exist_ok=True)

    def HECROP(SLIDE, DOWN, box):
        """The parent's he-raw.jpg is gone. It was level 0 divided by DOWN, so read the same box from the pyramidal TIFF."""
        import openslide
        from PIL import Image
        x0, y0, x1, y1 = box
        sl = openslide.OpenSlide(f"{R}/data/visiumhd/{SLIDE}/{SLIDE}_PYRAMIDAL.tif")
        im = sl.read_region((int(round(x0 * DOWN)), int(round(y0 * DOWN))), 0,
                            (int(round((x1 - x0) * DOWN)), int(round((y1 - y0) * DOWN)))).convert("RGB")
        return im.resize((x1 - x0, y1 - y0), Image.LANCZOS)

    run_patched(f"{R}/fig1_windows.py", [
        ('json.load(open(os.path.join(ROOT, "results/ruler_fold.json")))', f'json.load(open("{OUT}/t32/ruler_fold.json"))'),
        ('OUT = os.path.join(ROOT, "figures")', f'OUT = "{O}"'),
        ('os.path.join(ROOT, f"results/fig1_v2_candidates_{TAG}.json")', f'os.path.join("{O}", f"fig1_v2_candidates_{{TAG}}.json")'),
        ('os.path.join(ROOT, "results/fig1_v2_numbers.json")', f'os.path.join("{O}", "fig1_v2_numbers.json")'),
        ('crop = Image.open(HEP).crop(', 'crop = HECROP(SLIDE, DOWN, '),
    ], ["fig1_windows.py"], {"HECROP": HECROP})
elif task == "sweep":
    import tower_sweep
    sys.argv = ["tower_sweep.py"] + rest
    tower_sweep.main()
elif task == "buffer":
    run_patched(f"{R}/src/buffer_cv.py", [
        ('"/path/to/project/results/buffer_cv.json"', f'"{OUT}/buffer_cv.json"')], ["buffer_cv.py"])
else:
    raise SystemExit(f"unknown task {task}")
