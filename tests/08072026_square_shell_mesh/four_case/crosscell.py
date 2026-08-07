"""Square-lattice CROSS cell: mesh + input-deck builders shared by the three
routes.

GEOMETRY.  Unit cell [0, L] x [0, L] in (y2, y3).  ONE wall of each family runs
THROUGH the cell:

    * horizontal wall : along y2, centred on y3 = L/2, thickness t
    * vertical   wall : along y3, centred on y2 = L/2, thickness t

The two arms meet end-to-end across the periodic boundary, so a wall is t thick
in BOTH the shell and the solid route.  (A ring/annulus cell puts its walls ON
the cell boundary: the solid periodic tie then bonds neighbouring walls into a
single 2t wall while the shell keeps two independent t walls -- a t-independent
factor 4 on C44 alone.  That is why this file builds a cross, not a ring.)

Material area of the cross cell = 2*L*t - t^2 (the junction square is counted
once).  The SHELL idealization integrates each wall over its full length L, so
it counts the junction twice -> 2*L*t.  The difference is O(t/L) and is part of
the honest shell-vs-solid comparison; it is reported by `wall_area_report`.

Voigt / component conventions
-----------------------------
* yaml node rows and yaml `elementOrientations` components are ordered
  (y2, y3, axial); (y2, y3, axial) is a cyclic permutation of (axial, y2, y3)
  so it is right-handed and e3 = e1 x e2 may be taken componentwise.
* the solvers index global axes as (1, 2, 3) = (axial, y2, y3); the yaml rows
  are cycled by opensg_solid.sg_materials.elem_rotation_from_yaml.
* stiffness order for every route: [e11 e22 e33 2e23 2e13 2e12], axis 1 = the
  prismatic (out-of-plane) direction.
"""
import numpy as np

# ---------------------------------------------------------------- materials --
ISO = dict(name="iso", E=(70.0e9,)*3, nu=(0.30,)*3,
           G=(70.0e9/(2*(1+0.30)),)*3, angle=0.0, density=2700.0)
M45 = dict(name="m45", E=(142.0e9, 9.8e9, 9.8e9), G=(6.0e9, 6.0e9, 4.8e9),
           nu=(0.30, 0.30, 0.42), angle=-45.0, density=1800.0)

# material_param row for the solid engines: E1 E2 E3 G12 G13 G23 nu12 nu13 nu23
def material_param_row(m):
    return (m["E"][0], m["E"][1], m["E"][2],
            m["G"][0], m["G"][1], m["G"][2],
            m["nu"][0], m["nu"][1], m["nu"][2])


# ------------------------------------------------------------- shell (1-D) ---
def cross_shell_yaml(path, L, t, mat, nseg=80):
    """1-D shell SG of the cross cell: two crossing wall MIDLINES, 2-node line
    elements, `nseg` elements per wall, sharing the centre node.

    elementOrientations rows are [e1 | e2 | e3] with
        e1 = [0, 0, 1]           beam axis, OUT of the cross-section plane
        e2 = wall tangent        (+y2 horizontal wall, +y3 vertical wall)
        e3 = e1 x e2             wall normal
    reference: center (the midline is the wall mid-surface)."""
    assert nseg % 2 == 0
    c = 0.5*L
    xs = np.linspace(0.0, L, nseg+1)
    h_n = np.stack([xs, np.full(nseg+1, c)], 1)          # along y2 at y3 = c
    v_n = np.stack([np.full(nseg+1, c), xs], 1)          # along y3 at y2 = c
    ic = nseg//2                                          # the shared centre
    keep = [j for j in range(nseg+1) if j != ic]
    pts = np.vstack([h_n, v_n[keep]])
    vid, k = {}, len(h_n)
    for j in range(nseg+1):
        if j == ic:
            vid[j] = ic
        else:
            vid[j] = k
            k += 1
    cells = ([[i, i+1] for i in range(nseg)]
             + [[vid[j], vid[j+1]] for j in range(nseg)])
    ori = ([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]*nseg      # horiz
           + [[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0]]*nseg)  # vert

    o = ["nodes:"]
    for x, y in pts:
        o.append("- [%.12f %.12f 0.00000000]" % (x, y))
    o.append("elements:")
    for a_, b_ in cells:
        o.append("- [%d %d]" % (a_+1, b_+1))
    o.append("elementOrientations:")
    for r in ori:
        o.append("- [" + ", ".join("%.1f" % v for v in r) + "]")
    o += ["sections:", "- type: shell", "  elementSet: layup_0", "  layup:",
          "  - - %s" % mat["name"], "    - %.10e" % t,
          "    - %.4f" % mat["angle"],
          "materials:", "- name: %s" % mat["name"],
          "  density: %.1f" % mat["density"], "  elastic:",
          "    E: [%.8e, %.8e, %.8e]" % tuple(mat["E"]),
          "    G: [%.8e, %.8e, %.8e]" % tuple(mat["G"]),
          "    nu: [%.8f, %.8f, %.8f]" % tuple(mat["nu"]),
          "sets:", "  element:", "  - name: layup_0", "    labels:"]
    o += ["    - %d" % (e+1) for e in range(len(cells))]
    o.append("reference: center")
    open(path, "w").write("\n".join(o) + "\n")
    return path, len(pts), len(cells)


# ------------------------------------------------------------- solid (2-D) ---
def cross_solid_mesh(L, t, nt, na):
    """Linear-triangle mesh of the cross cell.

    nt = elements ACROSS the wall thickness (>= 6 required: CST locks in
    bending and C44 is the only bending-dominated term).
    na = elements ALONG each arm.  The along-wall size MUST scale with t --
    a fixed along-wall size makes the solid spuriously stiff as t shrinks.

    Returns (nodes (N,2), tri (E,3) 0-based, frame (E,) 0=horiz wall,
    1=vertical wall, area (E,)).  The junction square is split along its
    diagonals so the 90 deg symmetry of the cell is preserved for anisotropic
    plies: a triangle belongs to the wall whose centreline it is nearer to."""
    c = 0.5*L
    lo, hi = c - 0.5*t, c + 0.5*t
    xa = np.unique(np.concatenate([np.linspace(0.0, lo, na+1),
                                   np.linspace(lo, hi, nt+1),
                                   np.linspace(hi, L, na+1)]))
    ng = len(xa)
    inband = lambda a, b: abs(0.5*(a+b) - c) < 0.5*t
    qc = [(i, j) for i in range(ng-1) for j in range(ng-1)
          if inband(xa[i], xa[i+1]) or inband(xa[j], xa[j+1])]
    gid = -np.ones((ng, ng), int)
    nid = 0
    for (i, j) in qc:
        for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1)):
            if gid[i+di, j+dj] < 0:
                gid[i+di, j+dj] = nid
                nid += 1
    nd = np.zeros((nid, 2))
    for i in range(ng):
        for j in range(ng):
            if gid[i, j] >= 0:
                nd[gid[i, j]] = (xa[i], xa[j])
    tri = []
    for (i, j) in qc:
        n00, n10, n01, n11 = gid[i, j], gid[i+1, j], gid[i, j+1], gid[i+1, j+1]
        if (i + j) % 2 == 0:
            tri += [[n00, n10, n11], [n00, n11, n01]]
        else:
            tri += [[n00, n10, n01], [n10, n11, n01]]
    tri = np.array(tri, int)

    p1, p2, p3 = nd[tri[:, 0]], nd[tri[:, 1]], nd[tri[:, 2]]
    det = ((p2[:, 0]-p1[:, 0])*(p3[:, 1]-p1[:, 1])
           - (p3[:, 0]-p1[:, 0])*(p2[:, 1]-p1[:, 1]))
    area = 0.5*np.abs(det)
    cen = (p1 + p2 + p3)/3.0
    # nearer the vertical centreline (y2 = c) -> vertical wall
    frame = np.where(np.abs(cen[:, 0]-c) < np.abs(cen[:, 1]-c), 1, 0)
    return nd, tri, frame, area


# rows [e1 | e2 | e3] in yaml component order (y2, y3, axial)
FRAME_ROW = {0: [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
             1: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0]}


def cross_solid_yaml(path, nd, tri, frame, mat):
    o = ["nodes:"]
    for x, y in nd:
        o.append("- [%.12f %.12f 0.00000000]" % (x, y))
    o.append("elements:")
    for a_, b_, c_ in tri:
        o.append("- [%d %d %d]" % (a_+1, b_+1, c_+1))
    o.append("elementOrientations:")
    for f in frame:
        o.append("- [" + ", ".join("%.1f" % v for v in FRAME_ROW[int(f)]) + "]")
    o += ["materials:", "- name: %s" % mat["name"],
          "  density: %.1f" % mat["density"], "  elastic:",
          "    E: [%.8e, %.8e, %.8e]" % tuple(mat["E"]),
          "    G: [%.8e, %.8e, %.8e]" % tuple(mat["G"]),
          "    nu: [%.8f, %.8f, %.8f]" % tuple(mat["nu"]),
          "sets:", "  element:", "  - name: wall", "    labels:"]
    o += ["    - %d" % (e+1) for e in range(len(tri))]
    open(path, "w").write("\n".join(o) + "\n")
    return path


# ------------------------------------------------------------------ checks --
def wall_area_report(L, t, area):
    """(meshed, exact 2Lt-t^2, rel err, shell-counted 2Lt)."""
    exact = 2*L*t - t*t
    got = float(np.sum(area))
    return got, exact, (got-exact)/exact, 2*L*t


# ----------------------------------------------------------- mesh sizing -----
NT_PROD, ASPECT = 18, 1.5


def mesh_sizing(L, t, refine=1.0):
    """The MESHING RULE, as settled by convergence.py.

    C44 is the only BENDING-dominated term and CST triangles lock in bending.
    Two independent knobs matter and BOTH have to scale with t:

      nt -- elements THROUGH the wall thickness.  A CST cannot carry the
            linear-through-thickness axial strain of a bending wall, so the
            C44 error decays like ~1/nt.  nt = 18 (the bare 6 leaves +30%).
      na -- elements ALONG each arm, set from an element ASPECT RATIO of 1.5,
            i.e. na = arm/(1.5*t/nt).  This makes the along-wall size scale
            with t.  A size fixed in L instead makes the solid spuriously
            stiff as t shrinks (+30% at t/L = 0.01 with along-wall = t/2, and
            far worse with along-wall = L/30).

    refine scales both knobs together (the 1.5x convergence check)."""
    nt = int(round(NT_PROD*refine))
    arm = 0.5*L - 0.5*t
    na = int(round(arm*nt/(ASPECT*t)))
    return nt, na
