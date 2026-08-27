# CUDA petsc4py for `--solver gamg` (iter 4) on Colab

`--solver gamg` runs PETSc GAMG-preconditioned CG (sg_gamg, through
the vendored jetsci layer).  On CPU any petsc4py works
(`pip install petsc4py`, or the conda-forge package).  On a GPU
runtime the PETSc library itself must be built with CUDA -- there is
no prebuilt CUDA wheel -- so it is a one-time source build you cache
on Drive next to the `_env` the benchmark notebook already uses.

## One-time build (A100 / CUDA-12 runtime)

Run in a fresh GPU session, after mounting Drive (same `ROOT` as
`OpenSG_GPU_benchmark.ipynb`):

```bash
ENV=/content/drive/MyDrive/OpenSG-2.0/_env
export PETSC_CONFIGURE_OPTIONS="--with-cuda=1 --with-debugging=0 COPTFLAGS=-O3 CXXOPTFLAGS=-O3"
pip install --target=$ENV --upgrade petsc petsc4py
```

**Build-time warning: expect 25-45 minutes** (PETSc configure +
compile on the Colab VM).  It happens ONCE: the finished install
lives in the Drive `_env`, and because Drive always mounts at the
same path, the PETSc prefix petsc4py bakes in at build time stays
valid in every later session -- later sessions just put `$ENV` on
`sys.path` (the notebook's cell 1 already does) and import.

If `import petsc4py` in a later session cannot find PETSc, point it
explicitly before the import:

```python
import os
os.environ["PETSC_DIR"] = "/content/drive/MyDrive/OpenSG-2.0/_env/petsc"
```

## The GPU engages automatically

`sg_gamg` takes its PETSc type names from jetsci's option plumbing
(`jetsci.petsc_ksp.options.PETScMethodOptions`): whenever jax sees a
GPU the Mat type is `aijcusparse` -- jetsci's own default -- so a
CUDA petsc4py runs the CG matvecs and the GAMG cycles on the card
with no extra flag.  On CPU the same code path uses `aij`.
(petsc4py builds <= 3.21 predate the COO Mat api; sg_gamg detects
that and takes a CSR construction fallback automatically -- the pip
build above is newer and uses the jetsci COO recipe directly.)

## Timing check vs iter 2 (amg)

Same case, both iterative backends, one command (case yaml copied to
local disk first -- solve on `/content`, never on Drive FUSE):

```bash
for s in amg gamg; do
  python -m opensg_solid /content/case.yaml --solver $s | grep -E "solver|Time taken"
done
```

Both must print the same law digits (rtol `OPENSG_AMG_RTOL`, default
1e-8, shared by amg and gamg; the tolerance certificate is one rerun
at `OPENSG_AMG_RTOL=1e-10`).  `OPENSG_GAMG_VERBOSE=1` adds one line
per solve block with CG iteration counts and the worst relative
residual.
