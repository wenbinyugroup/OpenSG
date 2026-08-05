# `abaqus_inp.py` — the static cylindrical-bending strip deck of this case

```
layup_db.yaml  ->  1dsg.yaml (+ homo .out, via the core rm_homo)
               ->  <name>_static_S8R.inp
```

The deck is the ex5 shell-deck architecture with **three deliberate
deviations**, all driven by the `plate:` block of `layup_db.yaml` — nothing is
edited in code between cases:

| key | effect | why |
|---|---|---|
| `static: true` | `*STATIC` / `1.0, 1.0` replaces `*DYNAMIC, DIRECT` | Pagano (1969) and Yu §6.1 are static benchmarks |
| `load_mode: cyl` | `*DLOAD` carries `q = q0 sin(pi x/a)` only — no `sin(pi y/b)` factor | cylindrical bending: the load has no y-variation |
| (implied) | `NALL, 2, 2` and `NALL, 4, 4` on every node | **plane strain**: v = 0 and the twist rotation UR1 = 0 everywhere, so the strip bends cylindrically |

## The keywords, in order

- **`*NODE` / `*ELEMENT, TYPE=S8R`** — the ex5 half-index serendipity grid
  (corners + edge midpoints, no face centres). Edge NSETs are filtered from
  the grid, so midside nodes are constrained too.
- **`*NSET X0 / XA`** — the two supported edges (x = 0 and x = a), every node
  through the width including midsides.
- **`*SHELL GENERAL SECTION`** — the 21 upper-triangle terms of the
  homogenized 6×6 (column-major), from the core `rm_homo` run of
  `layup_db.yaml`. No plies in the model: the laminate exists only through
  this section law.
- **`*TRANSVERSE SHEAR STIFFNESS`** — the MSG 2×2 block (K11, K22, K12).
  S8R carries transverse shear (it prints SF4/SF5), so the card is live.
- **`*BOUNDARY`** — SS at both supported edges (`w = 0`), one axial anchor
  (`u = 0` at x = 0 — statically determinate, no membrane locking of the
  sine solution), then the plane-strain cards on ALL nodes plus the drilling
  fix (`NALL, 6, 6`; S8R is a 6-dof element on a flat mesh).
- **`*STATIC`** — one increment; there is no time in the problem.
- **`*DLOAD`** — `P` per element, `q0 sin(pi x_c/a)` at the element-centre x.
  For the Yu cases `q0` is the NET load p0 of the split face pair
  s3 = b3 = (p0/2) sin(px): a shell feels only the net transverse load; the
  split enters the recovery through the load ladder, not the deck.
- **`*NODE PRINT NALL / U` + `*EL PRINT EALL / SF, SM` + `COORD`** — the
  whole-strip field dump (one static increment), exactly the tables
  `rm_dehom_dat.py` parses. SF/SM and COORD are two separate blocks — merged
  they become one table and the parser's reshape breaks.

## What the deck is for

The analytical chain (`run_case.py`) does not need Abaqus: the FF of the
harmonic solution is closed-form. This deck exists so the SAME case can also
be run through a shell FE and its `.dat` fed to `rm_dehom_dat.py` — the
Abaqus-FF route of the pipeline — and the two FF sources compared. Run it
with:

```bash
abaqus job=<name>_static_S8R cpus=2 interactive
```
