"""run_bare_gpu.py -- the Bare-folder tet10 convergence sweep on a
Colab GPU, RESUMABLE: every completed law lands as <stem>.out in
OUT_DIR (put it on Drive), and a rerun after a session kill skips
everything already done.

Two jobs in one file:
  1. every tet4 .msh in MESH_DIR -> tet10 twin -> yaml (Al, plate,
     shear-refined) -> homogenization on the GPU -> <stem>_quad.out
  2. a solver sweep on ONE calibration case: every AVAILABLE solver
     option (direct = CPU pardiso/superlu for reference, cg = the GPU
     matrix-free route, stream if wired) timed on the same yaml, laws
     cross-checked -> solver_sweep.dat

    python run_bare_gpu.py --meshes /content/drive/MyDrive/meshes \
        --out /content/drive/MyDrive/OpenSG-2.0/colab_results

In:  --meshes DIR with the linear SP_solid_*.msh files (duplicates by
     content are skipped via size+md5); --out DIR on Drive for .out +
     tables; --work DIR local scratch (default /content/work);
     --skip-elems N skip meshes above N linear elements (default 5e6);
     --sweep-case STEM the calibration case (default the smallest)
Out: OUT_DIR/<stem>_quad.out per mesh, OUT_DIR/convergence_gpu.dat,
     OUT_DIR/solver_sweep.dat
"""
import argparse
import datetime
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import time

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--meshes", required=True)
p.add_argument("--out", required=True)
p.add_argument("--work", default="/content/work")
p.add_argument("--skip-elems", type=float, default=5e6)
p.add_argument("--sweep-case", default=None)
a = p.parse_args()

os.makedirs(a.out, exist_ok=True)
os.makedirs(a.work, exist_ok=True)
ENV = dict(os.environ)
ENV["JAX_ENABLE_X64"] = "1"
# the assembly slab must fit DEVICE memory next to the solver state:
# ~1 GB is right for a 16 GB T4 (override if on an A100)
ENV.setdefault("OPENSG_SLAB_BYTES", "1e9")
ENV.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax                                            # noqa: E402
print("start   :", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("devices :", jax.devices())
print("slab    :", ENV["OPENSG_SLAB_BYTES"], "bytes")


def law8(path):
    rows = []
    for ln in open(path):
        v = ln.split()
        if len(v) == 8:
            try:
                rows.append([float(x) for x in v])
            except ValueError:
                pass
        if len(rows) == 8:
            break
    return np.array(rows) if len(rows) == 8 else None


def sh(cmd, cwd):
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True,
                       text=True)
    dt = time.perf_counter() - t0
    tail = (r.stdout + r.stderr).strip().split("\n")[-4:]
    return r.returncode, dt, tail


# ---- job 1: every distinct linear mesh -> tet10 .out
seen, cases = {}, []
for m in sorted(glob.glob(os.path.join(a.meshes, "*.msh"))):
    if m.endswith("_quad.msh"):
        continue
    key = (os.path.getsize(m),
           hashlib.md5(open(m, "rb").read(1 << 20)).hexdigest())
    if key in seen:
        print("skip duplicate:", os.path.basename(m), "==",
              os.path.basename(seen[key]))
        continue
    seen[key] = m
    cases.append(m)

for m in cases:
    stem = os.path.splitext(os.path.basename(m))[0]
    out_law = os.path.join(a.out, stem + "_quad.out")
    if os.path.exists(out_law) and law8(out_law) is not None:
        print("done   :", stem, "(resumed, skipping)")
        continue
    ne = sum(1 for _ in open(m)) - 10          # cheap element-count proxy
    if ne > a.skip_elems * 1.6:
        print("skip   :", stem, "(too large for this session)")
        continue
    lm = os.path.join(a.work, os.path.basename(m))
    if not os.path.exists(lm):
        shutil.copy(m, lm)
    print("== %s  %s" % (stem,
                         datetime.datetime.now().strftime("%H:%M:%S")))
    rc, dt, tail = sh([sys.executable, "-m", "opensg", "msh_to_yaml",
                       os.path.basename(lm), "--mat1", "Al",
                       "--n_model", "2", "--refined", "1",
                       "--p_refine"], a.work)
    print("   convert %.0f s rc=%d" % (dt, rc))
    if rc:
        print("   " + "\n   ".join(tail))
        continue
    qy = os.path.basename(lm).replace(".msh", "_quad.yaml")
    rc, dt, tail = sh([sys.executable, "-m", "opensg", qy, "H",
                       "--solver", "cg"], a.work)
    print("   homo    %.0f s rc=%d" % (dt, rc))
    if rc:
        print("   " + "\n   ".join(tail))
        continue
    shutil.copy(os.path.join(a.work, qy.replace(".yaml", ".out")),
                out_law)
    print("   saved ->", out_law)
    for f in glob.glob(os.path.join(a.work, stem + "*")):
        if not f.endswith(".msh"):
            os.remove(f)

# ---- job 2: the solver sweep on one calibration case
cal = a.sweep_case
if cal is None and cases:
    cal = os.path.splitext(os.path.basename(
        min(cases, key=os.path.getsize)))[0]
if cal:
    qy = cal + "_quad.yaml"
    if not os.path.exists(os.path.join(a.work, qy)):
        lm = os.path.join(a.work, cal + ".msh")
        if not os.path.exists(lm):
            shutil.copy(os.path.join(a.meshes, cal + ".msh"), lm)
        sh([sys.executable, "-m", "opensg", "msh_to_yaml",
            os.path.basename(lm), "--mat1", "Al", "--n_model", "2",
            "--refined", "1", "--p_refine"], a.work)
    rows, base = [], None
    for s in ("direct", "superlu", "cg", "stream"):
        rc, dt, tail = sh([sys.executable, "-m", "opensg", qy, "H",
                           "--solver", s], a.work)
        if rc:
            print("sweep %-8s FAILED/unwired: %s" % (s, tail[-1][:80]))
            rows.append((s, np.nan, np.nan))
            continue
        L = law8(os.path.join(a.work, qy.replace(".yaml", ".out")))
        if base is None:
            base, d = (s, L), 0.0
        else:
            d = float(np.abs(L - base[1]).max()
                      / max(np.abs(base[1]).max(), 1e-30))
        print("sweep %-8s %8.1f s   rel diff vs %s: %.2e"
              % (s, dt, base[0], d))
        rows.append((s, dt, d))
    with open(os.path.join(a.out, "solver_sweep.dat"), "w") as f:
        f.write("# solver sweep on %s -- %s\n# devices: %s\n"
                % (cal, datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"), jax.devices()))
        f.write("# solver  wall_s  rel_law_diff\n")
        for s, t, d in rows:
            f.write("%-8s %10.2f %12.3e\n" % (s, t, d))

# ---- the convergence table from every .out present
outs = sorted(glob.glob(os.path.join(a.out, "*_quad.out")))
with open(os.path.join(a.out, "convergence_gpu.dat"), "w") as f:
    f.write("# tet10 GPU convergence -- %s\n# case  A11  A66  D11  D22"
            "  G11  G22\n" % datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"))
    for o in outs:
        L = law8(o)
        if L is None:
            continue
        f.write("%-40s %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e\n"
                % (os.path.basename(o), L[0, 0], L[2, 2], L[3, 3],
                   L[4, 4], L[6, 6], L[7, 7]))
print(open(os.path.join(a.out, "convergence_gpu.dat")).read())
print("end     :", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
