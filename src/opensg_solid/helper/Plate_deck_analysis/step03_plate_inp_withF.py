"""step03_plate_inp_withF.py -- the plate deck WITH the pressure-induced
in-plane term of the MSG plate law, and the measurement of that term.

    python -m opensg_solid.helper.Plate_deck_analysis.step03_plate_inp_withF \
           inject --inp <plate.inp> --fn <F_N> [--out <withF.inp>]
    python -m opensg_solid.helper.Plate_deck_analysis.step03_plate_inp_withF \
           measure --yaml <sg.yaml> --q <pressure> --workdir <dir> \
           [--src-dir <dir>]

WHY (condensed from the validated one-off, pressure_case
abaqus_plate_2d/gen_plate_pressure_withF.py).  `*Shell General Section`
implements N = A e, M = D k.  The MSG plate law of a PRESSURE-LOADED SG
carries one term more (Yu 2003 Eq. 45):

    N = A e + F_N ,      M = D k + F_M ,

where F_N, F_M are the resultants of the SG's own pressure LOAD COLUMN:
the cell's response to q at ZERO macro strain.  MOMENT needs no deck
change -- equilibrium and the SS natural BC are written in the TOTAL
resultants, so the F-less deck's SM already solves the right BVP; F_M
only corrects the curvature k = D^-1 (M - F_M), a station-file matter.
MEMBRANE is the deck's real gap: with edges free in their own normal
direction an F-less plate solves N = A e = 0, hence e = 0 -- it cannot
strain in-plane under pressure at all, while the true law gives
e = -A^-1 F_N.  An OUTWARD edge traction w = -F_N restores it, applied
as consistent nodal `*Cload` forces (an S8R edge's `*Dsload` side
numbering depends on connectivity; nodal forces are unambiguous and
their sum is checkable): a uniform w on a quadratic edge segment of
length L puts w L/6 on each corner and 2 w L/3 on the midside, corner
nodes shared by two segments get both shares, and the plate-corner
nodes (in EDGEX and EDGEY at once) get both an x and a y entry.  BCs,
section, pressure and output stay byte-identical; the gate is the
per-edge force sum, which must equal w * edge length.

`measure` produces F_N, F_M with NO plate analysis: a zero-strain
station .ff (u/theta zero, `0:` all zeros, qt6 = [q, 0, ...]) is staged
next to copies of the yaml/.out, one `opensg_solid <yaml> D --global`
run recovers the field (= the load column, since the macro strain is
zero), and the SG tet volumes integrate it:

    F_N_ab = sum S_ab V_e ,      F_M_ab = sum S_ab z_e V_e .

Regression note (TPMS SP_solid_rho0.3_n2.54857_quad, q = 7946.1 Pa):
F_N11 = F_N22 = -567.876 N/m, F_M11 = F_M22 = -90.908 N.

In:  a base plate .inp + F_N in N/m, negative = compressive (inject);
     the SG yaml + face pressure q [Pa] + a work dir (measure)
Out: the _withF.inp, one `*Cload` block inserted before `*Dload`,
     everything else byte-identical (inject); <yaml stem>_loadcol.dat
     with F_N11 F_N22 F_N12 F_M11 F_M22 F_M12 (measure)"""
import argparse
import os
import shutil
import subprocess
import sys
from collections import defaultdict

import numpy as np


# ------------------------------------------------------------- inject
def _parse_deck(src):
    """Nodes and node sets of an Abaqus deck.

    In:  src list of str -- the deck lines
    Out: (nodes {id: (x, y)}, sets {NAME: [node ids]}) -- nset names
         uppercased; `generate` nsets expanded."""
    nodes, sets, mode, cur, gen = {}, {}, None, None, False
    for ln in src:
        if ln.startswith("*") and not ln.startswith("**"):
            low = ln.lower()
            if low.startswith("*node") \
                    and not low.startswith("*node output"):
                mode, cur = "node", None
            elif low.startswith("*nset"):
                mode = "nset"
                cur = ln.split("nset=")[1].split(",")[0].strip().upper()
                gen = "generate" in low
                sets.setdefault(cur, [])
            else:
                mode = None
            continue
        if mode is None or ln.startswith("**"):
            continue
        v = ln.replace(",", " ").split()
        try:
            if mode == "node" and len(v) >= 3:
                nodes[int(v[0])] = (float(v[1]), float(v[2]))
            elif mode == "nset" and gen and 2 <= len(v) <= 3:
                first, last = int(v[0]), int(v[1])
                inc = int(v[2]) if len(v) == 3 else 1
                sets[cur] += list(range(first, last + 1, inc))
            elif mode == "nset":
                sets[cur] += [int(k) for k in v
                              if k.lstrip("-").isdigit()]
        except ValueError:
            continue
    return nodes, sets


def _edge_cload(nodes, sets, w):
    """Consistent nodal forces of the uniform outward traction w on the
    four plate edges (EDGEX at x = const loaded along x, EDGEY at
    y = const along y).

    Each nset must sit on exactly two constant-coordinate sides; each
    side's nodes must form one quadratic chain (odd count, corner-
    midside alternation by position).  Corner nodes shared by two
    segments accumulate both shares; the plate corners appear in both
    nsets and so get an x and a y entry.

    In:  nodes {id: (x, y)}; sets {name: ids}; w float [N/m] --
         outward-positive traction (= -F_N)
    Out: (cload {(node, dof): force}, gates [str] -- one per edge,
         sum vs w * edge length)."""
    cload = defaultdict(float)
    gates = []
    for setname, axis in (("EDGEX", 0), ("EDGEY", 1)):
        if setname not in sets:
            raise SystemExit("deck has no nset %s" % setname)
        ids = [i for i in sets[setname] if i in nodes]
        if not ids:
            raise SystemExit("nset %s has no known nodes" % setname)
        cs = [nodes[i][axis] for i in ids]
        smin, smax = min(cs), max(cs)
        if smax - smin <= 0.0:
            raise SystemExit("nset %s sits on one side only" % setname)
        tol = 1e-6 * (smax - smin) + 1e-12
        mid = 0.5 * (smin + smax)
        counted = 0
        for side in (smin, smax):
            line = sorted(
                [i for i in ids if abs(nodes[i][axis] - side) <= tol],
                key=lambda i: nodes[i][1 - axis])
            counted += len(line)
            if len(line) < 3 or len(line) % 2 == 0:
                raise SystemExit(
                    "%s %+g: %d nodes is not a quadratic edge chain"
                    % (setname, side, len(line)))
            t = [nodes[i][1 - axis] for i in line]
            sgn = 1.0 if side > mid else -1.0
            tot = 0.0
            for k in range(0, len(line) - 2, 2):
                L = t[k + 2] - t[k]
                for nd, share in ((line[k], 1.0 / 6),
                                  (line[k + 1], 2.0 / 3),
                                  (line[k + 2], 1.0 / 6)):
                    f = w * L * share * sgn
                    cload[(nd, axis + 1)] += f
                    tot += f
            target = w * (t[-1] - t[0]) * sgn
            gates.append("%s %+.4f: %d nodes, sum F%d = %+.3f N"
                         " (target %+.3f)"
                         % (setname, side, len(line), axis + 1, tot,
                            target))
        if counted != len(ids):
            raise SystemExit("%s: %d of %d nodes are on neither side"
                             % (setname, len(ids) - counted, len(ids)))
    return cload, gates


def inject(inp_path, F_N, out_path=None):
    """The _withF deck: one `*Cload` block before `*Dload`, everything
    else byte-identical.

    In:  inp_path str -- the base plate .inp; F_N float [N/m], the
         load-column resultant (negative = compressive; the applied
         traction is w = -F_N, outward); out_path str | None
         (default <inp stem>_withF.inp)
    Out: the path written; per-edge gate lines printed."""
    src = open(inp_path).read().splitlines()
    nodes, sets = _parse_deck(src)
    w = -float(F_N)
    cload, gates = _edge_cload(nodes, sets, w)

    block = ["** consistent nodal forces of a uniform outward in-plane"
             " edge traction",
             "**   w = %+.3f N/m on all four edges  (= -F_N)" % w,
             "*Cload"]
    for (nd, dof), f in sorted(cload.items()):
        block.append("%d, %d, %.6f" % (nd, dof, f))

    out, hit = [], False
    for ln in src:
        if not hit and ln.lower().startswith("*dload"):
            out += block
            hit = True
        out.append(ln)
    if not hit:
        raise SystemExit("no *Dload in %s -- nowhere to anchor the"
                         " *Cload block" % inp_path)

    if out_path is None:
        out_path = os.path.splitext(inp_path)[0] + "_withF.inp"
    open(out_path, "w", newline="\n").write("\n".join(out) + "\n")
    for g in gates:
        print("gate: " + g)
    print("wrote %s (added one *Cload block, %d entries; everything"
          " else byte-identical)" % (out_path, len(cload)))
    return out_path


# ------------------------------------------------------------ measure
def _write_zero_ff(path, q):
    """The zero-strain station file: dehom of THIS state is the load
    column itself (the cell response to q at zero macro strain).

    In:  path str -- <yaml stem>.ff; q float [Pa] -- TOP-face pressure
    Out: the path written."""
    with open(path, "w") as f:
        f.write("# zero-strain macro state -- step03 measure: the"
                " recovered field is the\n# SG load column of q (Yu"
                " 2005 Eq. 45 F_N/F_M source), no plate analysis\n")
        f.write("u: [0, 0, 0]\n")
        f.write("theta: [0, 0, 0]\n")
        f.write("0: [0, 0, 0, 0, 0, 0]\n")
        f.write("qt6: [%.10g, 0, 0, 0, 0, 0]\n" % float(q))
    return path


def _tet_geometry(npz_path):
    """Corner-tet volumes and centroid heights of the SG mesh sidecar.

    In:  npz_path str -- <yaml stem>_sg.npz (nodes, cells; cells may be
         1-based -- min == 1 subtracts 1; tet10 rows use their first 4
         corner nodes)
    Out: (V (E,) volumes, z (E,) corner-centroid z; z = 0 is the
         reference surface)."""
    d = np.load(npz_path, allow_pickle=True)
    nd = np.asarray(d["nodes"], float)[:, :3]
    cl = d["cells"]
    if isinstance(cl, np.ndarray) and cl.ndim == 2 \
            and cl.dtype != object:
        c4 = np.asarray(cl[:, :4], dtype=np.int64)
    else:
        c4 = np.asarray([np.asarray(c)[:4] for c in cl],
                        dtype=np.int64)
    if c4.min() == 1:
        c4 = c4 - 1
    a = nd[c4[:, 1]] - nd[c4[:, 0]]
    b = nd[c4[:, 2]] - nd[c4[:, 0]]
    c = nd[c4[:, 3]] - nd[c4[:, 0]]
    V = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
    z = nd[c4].mean(axis=1)[:, 2]
    return V, z


def measure(yaml_path, q, workdir, src_dir=None):
    """Measure F_N, F_M by one zero-strain dehom run + tet integration.

    Stages copies of the yaml (+ .out / _sg.npz when present) and the
    zero-strain .ff in workdir, runs `opensg_solid <yaml> D --global`
    there (OPENSG_SG_CACHE=1 so the _sg.npz sidecar exists for the
    volume integrals), then integrates the elemental global-frame .SM:
    F_N_ab = sum S_ab V_e, F_M_ab = sum S_ab z_e V_e, ab in {11,22,12}.

    In:  yaml_path str -- the SG yaml; q float [Pa] -- face pressure;
         workdir str; src_dir str | None -- dir holding the matching
         .out/_sg.npz (default: the yaml's own dir)
    Out: dict {F_N (3,), F_M (3,), dat} -- the _loadcol.dat path;
         values printed."""
    yaml_path = os.path.abspath(yaml_path)
    workdir = os.path.abspath(workdir)
    src = os.path.abspath(src_dir) if src_dir else \
        os.path.dirname(yaml_path)
    stem = os.path.splitext(os.path.basename(yaml_path))[0]
    os.makedirs(workdir, exist_ok=True)

    staged = []
    for name, always in ((os.path.basename(yaml_path), True),
                         (stem + ".out", False),
                         (stem + "_sg.npz", False)):
        s = yaml_path if always else os.path.join(src, name)
        t = os.path.join(workdir, name)
        if os.path.abspath(s) == t:
            staged.append(name)
            continue
        if os.path.exists(s):
            shutil.copy2(s, t)          # copy2: keeps the npz-vs-yaml
            staged.append(name)         # mtime freshness of the cache
    _write_zero_ff(os.path.join(workdir, stem + ".ff"), q)
    staged.append(stem + ".ff (zero strain, qt6 q = %g)" % float(q))
    print("staged %s: %s" % (workdir, ", ".join(staged)))

    cmd = [sys.executable, "-m", "opensg_solid",
           os.path.basename(yaml_path), "D", "--global"]
    log = os.path.join(workdir, stem + "_measure.log")
    env = dict(os.environ, OPENSG_SG_CACHE="1")
    with open(log, "w") as lf:
        rc = subprocess.call(cmd, cwd=workdir, stdout=lf,
                             stderr=subprocess.STDOUT, env=env)
    print("ran: %s (log %s)" % (" ".join(cmd[2:]), log))
    tail = open(log).read().splitlines()
    if rc != 0:
        for ln in tail[-15:]:
            print("  " + ln)
        raise SystemExit("opensg_solid exited %d" % rc)
    for ln in tail:
        if "recovery skipped" in ln:    # classical yaml has no load
            print("WARNING: " + ln)     # ladder -> the field is NOT
                                        # the load column

    sm = os.path.join(workdir, stem + "_dehom_elemental_global.SM")
    if not os.path.exists(sm):
        raise SystemExit("missing %s -- see %s" % (sm, log))
    S = np.atleast_2d(np.loadtxt(sm))[:, 3:9]   # S11 S22 S33 S12 ...

    npz = os.path.join(workdir, stem + "_sg.npz")
    if not os.path.exists(npz):
        npz = os.path.join(src, stem + "_sg.npz")
    if not os.path.exists(npz):
        raise SystemExit("no %s_sg.npz in workdir or src dir" % stem)
    V, z = _tet_geometry(npz)
    if len(S) != len(V):
        raise SystemExit("elemental .SM rows %d != %d SG tets"
                         % (len(S), len(V)))

    idx = np.asarray([0, 1, 3], dtype=np.int64)    # 11 22 12
    F_N = (S * V[:, None]).sum(axis=0)[idx]
    F_M = (S * (V * z)[:, None]).sum(axis=0)[idx]

    dat = os.path.join(workdir, stem + "_loadcol.dat")
    with open(dat, "w") as f:
        f.write("# SG pressure load column (Yu 2003 Eq. 45), q = %g"
                " Pa, zero macro strain\n" % float(q))
        f.write("# F_N11[N/m] F_N22[N/m] F_N12[N/m] F_M11[N]"
                " F_M22[N] F_M12[N]\n")
        f.write(" ".join("%+.6e" % v
                         for v in list(F_N) + list(F_M)) + "\n")
    print("F_N [N/m] 11 22 12: %+.3f %+.3f %+.4f" % tuple(F_N))
    print("F_M [N]   11 22 12: %+.3f %+.3f %+.4f" % tuple(F_M))
    print("wrote %s" % dat)
    return {"F_N": F_N, "F_M": F_M, "dat": dat}


def main(argv=None):
    """CLI: the two subcommands.

    In:  argv list | None
    Out: int exit status."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd")
    pi = sub.add_parser("inject", help="insert the -F_N edge traction"
                                       " *Cload block")
    pi.add_argument("--inp", required=True)
    pi.add_argument("--fn", type=float, required=True,
                    help="F_N [N/m], negative = compressive")
    pi.add_argument("--out", default=None)
    pm = sub.add_parser("measure", help="measure F_N/F_M from one"
                                        " zero-strain dehom run")
    pm.add_argument("--yaml", required=True)
    pm.add_argument("--q", type=float, required=True,
                    help="TOP-face pressure [Pa]")
    pm.add_argument("--workdir", required=True)
    pm.add_argument("--src-dir", default=None,
                    help="dir with the matching .out/_sg.npz"
                         " (default: the yaml's dir)")
    a = p.parse_args(argv)
    if a.cmd == "inject":
        inject(a.inp, a.fn, a.out)
    elif a.cmd == "measure":
        measure(a.yaml, a.q, a.workdir, a.src_dir)
    else:
        p.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
