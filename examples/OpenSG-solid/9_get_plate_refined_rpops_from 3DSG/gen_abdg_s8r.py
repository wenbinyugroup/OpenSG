"""gen_abdg_s8r.py -- the S8R (quadratic serendipity) equivalent-plate
twins of the AR5 ABDG S4R decks: same footprint (0..5 x 0..5,
reference surface = the SG mid-plane z = 0), same all-round clamp
(ENCASTRE -- corner AND edge-midside nodes), same q = 1 MPa down, same
*Shell General Section + *Transverse Shear Stiffness law from
preovios_try.out, LINEAR.  Two decks:

  preovios_try_ABDG8_AR5.inp    5 x 5 S8R, CELL-MATCHED (one element
                                per 3-D SG cell), job AR5_ABDG8
  preovios_try_ABDG8F_AR5.inp   25 x 25 S8R (5 per cell), the
                                cross-check twin, job AR5_ABDG8F

WHY S8R.  The displacement is quadratic, so the section forces vary
LINEARLY inside an element: the second derivatives of the strain
measures become resolvable at the cell-matched granularity itself,
without the subdivided mesh the S4R route needed (AR5_ABDGF).

Element ids keep the S4R generators' grid rule e = j*NE + i + 1 at
pitch a/NE, and the S8R dump (abq_dump_plate_s8r.py) averages the 4
integration points to ONE row per element, so build_ff.ff_grid parses
the reports unchanged.  Node ids are free (nobody reads them by
formula); the 8-node connectivity is Abaqus S8R order: corners 1-4
CCW (+z normal), then midsides 5 (1-2), 6 (2-3), 7 (3-4), 8 (4-1).

RESULTS ROUTING (deliberate).  claude_tmp\\S8R5.bat copies the job
output to abaqus/AR5_S8R/, NOT abaqus/AR5/: build_ff.py auto-discovers
<CASE>_ABDG*_SF.rpt inside abaqus/AR5/, AR5_ABDG8*_SF.rpt MATCHES that
glob, and build_ff keeps the LARGEST grid it finds -- dropped into
abaqus/AR5/, the 25x25 S8R run would silently displace AR5_ABDGF as
the derivative source.  Keep the S8R results in their own subfolder.

Each written deck is re-parsed and validated: node/element counts, the
element-id grid rule + S8R corner/midside ordering checked against the
coordinates, and the clamped nset checked geometrically BOTH ways
(every geometric boundary node in EDGE, every EDGE node on the
boundary), the way the C3D10 deck's midside audit was done.

In:  preovios_try.out
Out: preovios_try_ABDG8_AR5.inp, preovios_try_ABDG8F_AR5.inp
"""
import os
import re
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CELL = 1.0                   # the SG's in-plane period (unit cell)
NC = 5                       # AR5: 5 x 5 cells, plate 5 x 5
PRESS = -1.0                 # q = 1 MPa DOWN on a +z-normal shell
A = NC * CELL
DECKS = [(5, "ABDG8", "AR5_ABDG8"), (25, "ABDG8F", "AR5_ABDG8F")]
TOL = 1e-9

print("gen_abdg_s8r: start %s"
      % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

out = os.path.join(HERE, "preovios_try.out")
rows = []
for ln in open(out):
    v = ln.split()
    if len(v) == 8 and re.match(r"^-?\d", v[0]):
        try:
            rows.append([float(x) for x in v])
        except ValueError:
            pass
    if len(rows) == 8:
        break
K = np.array(rows)
if K.shape != (8, 8):
    raise SystemExit("%s is not an 8x8 shear-refined plate law" % out)
ABD = 0.5 * (K[:6, :6] + K[:6, :6].T)
G = 0.5 * (K[6:, 6:] + K[6:, 6:].T)
cpl = np.abs(ABD[:3, 3:]).max() / np.sqrt(ABD[0, 0] * ABD[3, 3])
print("law: A11 %.5e  D11 %.5e  G11 %.5e  G22 %.5e  |B|/sqrt(A11 D11)"
      " = %.2e" % (ABD[0, 0], ABD[3, 3], G[0, 0], G[1, 1], cpl))
if cpl > 2e-2:
    raise SystemExit("SG not centred -- this deck would misplace the law")

for NE, tag, job in DECKS:
    pit = A / NE                       # element pitch a/NE
    h = 0.5 * pit                      # half-pitch (midside spacing)
    m = 2 * NE + 1                     # half-step grid, per side
    nid, nn = {}, 0
    for gj in range(m):                # serendipity: no interior node
        for gi in range(m):
            if gi % 2 == 1 and gj % 2 == 1:
                continue
            nn += 1
            nid[(gi, gj)] = nn
    D = ["*Heading",
         "** AR5 %s equivalent plate of preovios_try.out -- %d x %d"
         " S8R (%s), clamped all round, LINEAR"
         % (tag, NE, NE, "cell-matched: one element per 3-D SG cell"
            if NE == NC else "5 per cell, cross-check twin"),
         "** reference surface = the SG mid-plane, z = 0",
         "** results -> abaqus/AR5_S8R/ (NOT AR5/: build_ff's ABDG*"
         " glob would pick these up)",
         "*Preprint, echo=NO, model=NO, history=NO, contact=NO",
         "*Node"]
    for gj in range(m):
        for gi in range(m):
            if (gi, gj) in nid:
                D.append("%d, %.6f, %.6f, 0."
                         % (nid[(gi, gj)], h * gi, h * gj))
    D.append("*Element, type=S8R")                          # CCW -> +z
    e = 0
    for j in range(NE):
        for i in range(NE):
            e += 1
            gi, gj = 2 * i, 2 * j
            cn = (nid[(gi, gj)], nid[(gi + 2, gj)],
                  nid[(gi + 2, gj + 2)], nid[(gi, gj + 2)],
                  nid[(gi + 1, gj)], nid[(gi + 2, gj + 1)],
                  nid[(gi + 1, gj + 2)], nid[(gi, gj + 1)])
            D.append("%d, %d, %d, %d, %d, %d, %d, %d, %d" % ((e,) + cn))
    edge = sorted(nid[k] for k in nid
                  if k[0] in (0, 2 * NE) or k[1] in (0, 2 * NE))
    D += ["*Elset, elset=ALL, generate", "1, %d, 1" % e,
          "*Nset, nset=EDGE"]
    for kk in range(0, len(edge), 16):
        D.append(", ".join(str(v) for v in edge[kk:kk + 16]))
    vals = [ABD[a, b] for b in range(6) for a in range(b + 1)]
    D.append("*Shell General Section, elset=ALL")
    for kk in range(0, 21, 8):
        D.append(", ".join("%.7e" % v for v in vals[kk:kk + 8]))
    D.append("*Transverse Shear Stiffness")
    D.append("%.7e, %.7e, %.7e" % (G[0, 0], G[1, 1], G[0, 1]))
    D += ["*Boundary", "EDGE, ENCASTRE",
          "*Step, name=STATIC, nlgeom=NO",
          "*Static",
          "*Dload", "ALL, P, %.4f" % PRESS,
          "*Output, field",
          "*Node Output"]
    D += (["U, UR, RF, RM"] if NE == NC else ["U, UR"])
    D += ["*Element Output, directions=YES"]
    D += (["SF, SM, SE"] if NE == NC else ["SF, SM"])
    D += ["*End Step", ""]
    p = os.path.join(HERE, "preovios_try_%s_AR5.inp" % tag)
    open(p, "w").write("\n".join(D))
    print("wrote %s : %d S8R (%d x %d), %d nodes, %d clamped edge"
          " nodes, plate %g x %g at z = 0, job %s"
          % (os.path.basename(p), e, NE, NE, nn, len(edge), A, A, job))

    # ---- VALIDATE by re-parsing the written deck ----
    nodes, elems, ns, mode = {}, {}, [], None
    for s in open(p):
        s = s.rstrip("\n")
        if s.startswith("*"):
            u = s.upper()
            if u.startswith("*NODE") and "OUTPUT" not in u:
                mode = "n"
            elif u.startswith("*ELEMENT,"):
                mode = "e"
            elif u.startswith("*NSET") and "EDGE" in u:
                mode = "s"
            else:
                mode = None
            continue
        if mode == "n":
            v = s.split(",")
            nodes[int(v[0])] = (float(v[1]), float(v[2]))
        elif mode == "e":
            v = [int(x) for x in s.split(",")]
            elems[v[0]] = v[1:]
        elif mode == "s":
            ns += [int(x) for x in s.split(",") if x.strip()]
    exp_n = (3 * NE + 1) * (NE + 1)
    assert len(nodes) == exp_n, "node count %d != %d" % (len(nodes),
                                                         exp_n)
    assert len(elems) == NE * NE, "element count"
    assert sorted(elems) == list(range(1, NE * NE + 1)), "element ids"
    for ee, cn in elems.items():
        assert len(cn) == 8, "element %d is not 8-node" % ee
        i, j = (ee - 1) % NE, (ee - 1) // NE      # THE grid rule
        x0, y0 = i * pit, j * pit
        c = [nodes[k] for k in cn]
        for k, (xe, ye) in enumerate([(x0, y0), (x0 + pit, y0),
                                      (x0 + pit, y0 + pit),
                                      (x0, y0 + pit)]):
            assert abs(c[k][0] - xe) < TOL and abs(c[k][1] - ye) < TOL, \
                "element %d corner %d off its grid cell" % (ee, k + 1)
        for k, (a_, b_) in enumerate([(0, 1), (1, 2), (2, 3), (3, 0)]):
            mx = 0.5 * (c[a_][0] + c[b_][0])
            my = 0.5 * (c[a_][1] + c[b_][1])
            assert abs(c[4 + k][0] - mx) < TOL \
                and abs(c[4 + k][1] - my) < TOL, \
                "element %d node %d is not the %d-%d midpoint" \
                % (ee, 5 + k, a_ + 1, b_ + 1)
    geo = {n for n, (x, y) in nodes.items()
           if min(x, y) < 1e-6 or max(x, y) > A - 1e-6}
    assert len(ns) == len(set(ns)), "EDGE nset has duplicates"
    assert set(ns) == geo, "EDGE nset != geometric boundary set" \
        " (nset-only %s, geo-only %s)" % (sorted(set(ns) - geo)[:5],
                                          sorted(geo - set(ns))[:5])
    mids = sum(1 for n in ns
               if any(int(round(nodes[n][d] / h)) % 2 == 1
                      for d in (0, 1)))
    assert len(ns) == 8 * NE and mids == 4 * NE, "edge census"
    txt = open(p).read()
    for req in ("*Element, type=S8R", "*Shell General Section,"
                " elset=ALL", "*Transverse Shear Stiffness",
                "EDGE, ENCASTRE", "*Step, name=STATIC, nlgeom=NO",
                "*Static", "*Dload", "ALL, P, %.4f" % PRESS,
                "*Output, field", "SF, SM"):
        assert req in txt, "deck is missing '%s'" % req
    print("  VALIDATED: %d nodes, %d S8R, all %d elements on the"
          " e = j*NE+i+1 grid with S8R corner/midside ordering;"
          " EDGE = geometric boundary both ways: %d nodes = 4 corners"
          " + %d edge midsides + %d edge corner-line nodes; section/"
          "TSS/clamp/Dload/step lines present; %d bytes"
          % (len(nodes), len(elems), len(elems), len(ns), mids,
             len(ns) - 4 - mids, os.path.getsize(p)))

print("gen_abdg_s8r: end %s"
      % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
