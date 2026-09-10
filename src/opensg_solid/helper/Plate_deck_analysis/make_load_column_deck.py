"""make_load_column_deck.py -- measure a plate's load-column terms and build
the equivalent-load deck, for ANY SG, ANY plate .inp and ANY pressure.

    python -m opensg_solid.helper.Plate_deck_analysis.make_load_column_deck \
           --inp <plate.inp> --src-dir <SG folder> --q <Pa> \
           [--workdir <tmp>] [--out <deck.inp>] [--with-moment] [--reuse <dat>]

THE THEORY IT IMPLEMENTS.  Yu, Hodges & Volovoi, Comput. Struct. 81 (2003)
439-454 splits the first-order warping as V1 = V11 e,1 + V12 e,2 + V1L
(their Eq. 45), the last column being the response to the APPLIED LOAD at
zero macro strain.  Carrying it through to the Reissner-like energy
(their Eq. 61)

    2 Pi_R = R^T A R + gamma^T G gamma + 2 R^T F

makes the constitutive law {N; M} = [ABD] {eps; kappa} + F, i.e.

    N = A eps + B kappa + F_N ,      M = B^T eps + D kappa + F_M .

Abaqus's `*Shell General Section` implements the law WITHOUT F.  The same
paper says the fix explicitly: "One must slightly modify traditional
Reissner-like plate solvers to accommodate these terms ... a form similar
to terms that must be included when considering thermal effects or actuated
materials."  An equivalent edge load is that modification.

HOW F IS OBTAINED.  Straight from Eq. 47, which for a load that does not
vary in plane (L,1 = L,2 = 0, V1L,1 = V1L,2 = 0) collapses to

    F = V0^T L        with   L = S+^T s - S-^T b - <S^T phi>

a single matrix product on quantities the homogenization has already
formed -- no dehomogenization run and no stress integration.  L is the
consistent nodal load of the applied tractions (the code's f_faces,
exported as r["L_faces"]); V0^T L is stored per unit face pressure as
r["F_unit"], and sg_dehom.load_column_F scales and combines the faces.

So a different unit cell, relative density, material or pressure gives its
own F with no constant to edit.  (For the Schwarz-P rho = 0.3 cell at
q = 7946.1 Pa this returns F_N11 = F_N22 = -567.876 N/m and
F_M11 = F_M22 = -90.923 N.)

MEMBRANE (default, and the one that works).  The deck's in-plane problem
carries no load at all, so it returns eps = 0 identically.  An outward edge
traction w = -F_N supplies it, as consistent nodal `*Cload` forces: on a
quadratic edge segment of length L the work-equivalent shares are L/6,
2L/3, L/6, so each edge sums to w * (edge length) -- printed as a gate.

MOMENT (--with-moment, OFF by default -- read this before using it).
The same weak-form argument makes an edge MOMENT m = -F_M on the free
bending rotation the EXACT equivalent of the F_M term, and it is applied
that way here (EDGEX -> DOF 5, EDGEY -> DOF 4, antisymmetric, opposite
family signs).  On the 5 x 5 TPMS plate it shifts the interior D.kappa by
+39.85 N for m = 90.908 -- not 90.908, because w = 0 on the boundary
resists a uniform curvature.  That is the correct Yu answer, not a
shortfall of the device.  It is OFF by default for an empirical reason
recorded in FINAL/NEGATIVE_RESULTS.md: the Yu-exact chain with both
terms puts the centre-cell total moment at -1710.5 N against the 3-D
reference -1758.9, WORSE than the F_N-only chain (-1750.4), and the
recovered ring sigma22 error rises accordingly.  The 3-D cell's own
M - D.kappa is -158 N, not the -91 N the SG law predicts, so the
residual is a limitation of the RM moment law at a/h = 5 (U* = 0.155),
not of the equivalent load.  Use it when the constitutive law is trusted
for the moment slot; measure before trusting.

In:  a plate .inp with nsets EDGEX (x = const) and EDGEY (y = const), an SG
     folder (yaml + .out + _sg.npz), the pressure q [Pa]
Out: <stem>_withF.inp (or _withFM.inp with --with-moment), one `*Cload`
     block inserted before `*Dload`, everything else byte-identical;
     the measured F and one gate line per edge printed
"""
import argparse
import os
import sys
from collections import defaultdict

from opensg_solid.helper.Plate_deck_analysis.step03_plate_inp_withF import (
    _edge_cload, _parse_deck)


def edge_moment_cload(nodes, sets, m):
    """Consistent nodal moments of a uniform edge moment m [N].

    DOF and sign are the ones measured on a probe of all eight
    combinations (probe_moment_dof.py): EDGEX takes DOF 5 (rotation about
    x2, conjugate to M11 on an x-normal face) and EDGEY takes DOF 4, both
    ANTISYMMETRIC between the two sides of the plate -- and the two
    families need OPPOSITE overall sign, or they cancel to a few per cent
    of the intended value.

    In:  nodes {id: (x, y)}; sets {name: ids}; m float [N] -- applied
         edge moment per unit length (= -F_M)
    Out: (cload {(node, dof): moment}, gates [str])."""
    cl, gates = defaultdict(float), []
    for setname, axis, dof, fam in (("EDGEX", 0, 5, +1.0),
                                    ("EDGEY", 1, 4, -1.0)):
        if setname not in sets:
            raise SystemExit("deck has no nset %s" % setname)
        ids = [i for i in sets[setname] if i in nodes]
        cs = [nodes[i][axis] for i in ids]
        smin, smax = min(cs), max(cs)
        if smax - smin <= 0.0:
            raise SystemExit("nset %s sits on one side only" % setname)
        tol = 1e-6 * (smax - smin) + 1e-12
        mid = 0.5 * (smin + smax)
        for side in (smin, smax):
            line = sorted([i for i in ids
                           if abs(nodes[i][axis] - side) <= tol],
                          key=lambda i: nodes[i][1 - axis])
            if len(line) < 3 or len(line) % 2 == 0:
                raise SystemExit("%s %+g: %d nodes is not a quadratic"
                                 " edge chain" % (setname, side, len(line)))
            t = [nodes[i][1 - axis] for i in line]
            s = fam * (1.0 if side > mid else -1.0)
            tot = 0.0
            for k in range(0, len(line) - 2, 2):
                L = t[k + 2] - t[k]
                for nd, sh in ((line[k], 1.0 / 6), (line[k + 1], 2.0 / 3),
                               (line[k + 2], 1.0 / 6)):
                    f = m * L * sh * s
                    cl[(nd, dof)] += f
                    tot += f
            gates.append("moment %s %+.4f: %d nodes, sum M%d = %+.4f N.m"
                         "  (target %+.4f)"
                         % (setname, side, len(line), dof, tot,
                            m * (t[-1] - t[0]) * s))
    return cl, gates


def build(inp_path, F_N, F_M=None, out_path=None):
    """Insert the equivalent-load `*Cload` block before `*Dload`.

    In:  inp_path str; F_N float [N/m]; F_M float [N] | None -- None
         means membrane only; out_path str | None
    Out: (path written, gates [str])."""
    src = open(inp_path).read().splitlines()
    nodes, sets = _parse_deck(src)
    w = -float(F_N)
    cl, gates = _edge_cload(nodes, sets, w)
    head = ["** equivalent load for the SG load column, Yu/Hodges/Volovoi",
            "** Comput. Struct. 81 (2003) 439-454, Eqs. 45, 47, 61:",
            "**   N = A eps + B kappa + F_N ,  M = B^T eps + D kappa + F_M",
            "**   in-plane edge traction  w = %+.6f N/m  (= -F_N)" % w]
    if F_M is not None:
        m = -float(F_M)
        cl_m, g_m = edge_moment_cload(nodes, sets, m)
        for k, v in cl_m.items():
            cl[k] += v
        gates += g_m
        head.append("**   edge moment            m = %+.6f N    (= -F_M)"
                    % m)
        head.append("**   NOTE: the moment device under-delivers on a"
                    " w = 0 boundary; see the module docstring")
    block = head + ["*Cload"] + ["%d, %d, %.6f" % (nd, d, f)
                                 for (nd, d), f in sorted(cl.items())]
    out, hit = [], False
    for ln in src:
        if not hit and ln.lower().startswith("*dload"):
            out += block
            hit = True
        out.append(ln)
    if not hit:
        raise SystemExit("no *Dload in %s -- nowhere to anchor the *Cload"
                         " block" % inp_path)
    if out_path is None:
        out_path = os.path.splitext(inp_path)[0] + (
            "_withFM.inp" if F_M is not None else "_withF.inp")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    open(out_path, "w", newline="\n").write("\n".join(out) + "\n")
    return out_path, gates


def main(argv=None):
    """CLI -- see the module docstring.

    In:  argv list | None.  Out: int exit status."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--inp", required=True, help="base plate .inp")
    p.add_argument("--src-dir", help="SG folder (yaml + .out + _sg.npz);"
                                     " omit only with --reuse")
    p.add_argument("--q", type=float, help="pressure amplitude [Pa]")
    p.add_argument("--reuse", help="a previously written _loadcol.dat,"
                                   " to skip the homogenization")
    p.add_argument("--with-moment", action="store_true",
                   help="also apply the F_M edge moment (read the"
                        " docstring: it under-delivers on a w = 0 edge)")
    p.add_argument("--out")
    a = p.parse_args(argv)

    if a.reuse:
        import numpy as np
        v = np.loadtxt(a.reuse)
        F_N, F_M = v[:3], v[3:]
        print("reused %s" % a.reuse)
    else:
        # Yu Eq. 47 with a load that does not vary in plane: one matrix
        # product on quantities the homogenization already has.
        if not (a.src_dir and a.q):
            raise SystemExit("--src-dir and --q are required without --reuse")
        import glob
        from opensg_solid.sg_dehom import load_column_F
        from opensg_solid.sg_homo import plate_homo_2d
        ys = glob.glob(os.path.join(a.src_dir, "*.yaml"))
        if len(ys) != 1:
            raise SystemExit("expected exactly one .yaml in %s, found %d"
                             % (a.src_dir, len(ys)))
        r = plate_homo_2d(ys[0], recovery=True)
        lc = load_column_F(r, q_top=float(a.q))
        F_N, F_M = lc["F_N"], lc["F_M"]
        print("Yu 2003 Eq. 47:  F = V0^T L   (L,a = V1L,a = 0, uniform q)")

    print("F_N [N/m] 11 22 12: %+.4f %+.4f %+.4f" % tuple(F_N))
    print("F_M [N]   11 22 12: %+.4f %+.4f %+.4f" % tuple(F_M))
    if abs(F_N[0] - F_N[1]) > 1e-3 * max(abs(F_N[0]), 1e-30):
        print("WARNING: F_N11 != F_N22; this builder applies the 11 value"
              " on both edge families")
    out, gates = build(a.inp, float(F_N[0]),
                       float(F_M[0]) if a.with_moment else None,
                       out_path=a.out)
    for g in gates:
        print("gate: " + g)
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
