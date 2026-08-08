"""
make_cylinder_segment.py
========================================================================
Generate a SURFACE (quad) shell mesh of a PRISMATIC cylinder, to drive the
3D-SG *tapered-segment* MITC-Reissner-Mindlin pipeline.

WHY A SURFACE MESH (and not the usual 1-D cross-section ring)?
--------------------------------------------------------------
The classical MSG thin-walled beam analysis meshes only the cross-section
CONTOUR (a 1-D ring) and enforces periodicity along the beam axis.  The
3D-SG *segment* method (IMECE2025 paper + OpenSG `mesh/segment.py`) instead
meshes a SLICE of the beam as a 2-D surface and treats aperiodicity/taper by:

  1. extracting the two END cross-sections (left/right rings) as separate,
     lower-dimensional boundary SGs  (dolfinx `create_submesh`);
  2. solving each boundary SG on its own for its warping fields V0 (Euler-
     Bernoulli) and V1 (Timoshenko shear-warping);
  3. transferring those boundary V0/V1 onto the segment's boundary DOFs as
     DIRICHLET constraints (this replaces the periodic BC and automatically
     fixes the segment's rigid-body modes);
  4. solving the segment, then recovering the Timoshenko 6x6.

For a PRISMATIC cylinder the two ends are identical and the taper (dx/dx1)
terms vanish, so the segment 6x6 MUST equal the single cross-section 6x6.
That identity is our validation self-check before moving to real (tapered)
blade segments.

THE TWO TEST CASES (the single-cell tubes we already validated in 1-D):
-----------------------------------------------------------------------
  * ISOTROPIC    : single ply [0], E = 70 GPa, G = 26.923 GPa, nu = 0.3
                   (matches cylinder_study/make_shell_iso.py)
  * ANISOTROPIC  : balanced [45/-45] (outer +45, inner -45),
                   E1 = 37, E2 = E3 = 9 GPa, G = 4 GPa, nu = 0.3
                   (matches aniso_tube/make_shell_aniso.py)

GEOMETRY / CONVENTIONS
----------------------
  * Beam axis          = x  (coordinate 0)  -- REQUIRED by segment.py, which
                         locates the end rings via x == x_min / x == x_max.
  * Cross-section plane = (y, z).
  * Reference surface   = wall MID-surface, radius R = 1.0 m  (center ref:
                         ABD B-block ~ 0 for the symmetric iso ply; the
                         [45/-45] coupling shows up as a genuine B16/B26).
  * Wall thickness      = t = hR * R.
  * NC quads around the hoop, NL quads along the axis over length L.
  * Per-element frame stored in `elementOrientations` as the 9-component
    row-major triad  [e1(3), e2(3), e3(3)]  with
        e1 = axial       = ( 1,  0,        0      )
        e2 = hoop tangent = ( 0, -sin th,  cos th )
        e3 = inward normal= ( 0, -cos th, -sin th )   (points to the axis)
    (Same e1/e2/e3 roles as the validated 1-D generators, rotated so the
     beam axis is x instead of z.)

OUTPUT: OpenSG-style YAML (nodes / elements(quad) / sections / sets /
materials / elementOrientations), read by the dolfinx extractor stage.
"""

import os
import math
import yaml

# ---------------------------------------------------------------- parameters
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes")
os.makedirs(OUT, exist_ok=True)

R = 1.0            # wall mid-surface radius [m]
NC = 160           # circumferential quads (== the 1-D ring resolution we tested)
NL = 3             # axial quads over the segment (>=2 so an interior ring exists)
L = 1.5            # segment length [m] (prismatic => value is immaterial to 6x6)
HR = [0.05, 0.1, 0.2]   # wall/radius ratios: thin, moderate, thick

# material cards (engineering constants, orthotropic E[3]/G[3]/nu[3])
ISO = {"name": "iso", "density": 1800.0,
       "E": [70e9, 70e9, 70e9], "G": [26.923e9, 26.923e9, 26.923e9], "nu": [0.3, 0.3, 0.3]}
ANI = {"name": "ani", "density": 1800.0,
       "E": [37e9, 9e9, 9e9], "G": [4e9, 4e9, 4e9], "nu": [0.3, 0.3, 0.3]}


def build_segment(material, layup_of_t, tag, hR):
    """Return the OpenSG mesh dict for one prismatic-cylinder surface segment.

    Parameters
    ----------
    material   : dict   material card (see ISO/ANI above)
    layup_of_t : callable(t) -> list of [mat_name, ply_thickness, angle_deg]
                 the stack for a wall of total thickness t (outer ply first)
    tag        : str    short label ("iso" / "aniso") used in the file name
    hR         : float  wall/radius ratio -> t = hR * R
    """
    t = hR * R
    dx = L / NL

    # ---- nodes: ring j (axial station x_j) x hoop node k (angle th_k) --------
    # node id (0-based) = j*NC + k ; stored 1-indexed in `elements`.
    # The hoop is CLOSED: node k = NC-1 is adjacent to node k = 0 (no dup node).
    nodes = []
    for j in range(NL + 1):
        xj = j * dx
        for k in range(NC):
            th = 2.0 * math.pi * k / NC
            nodes.append([xj, R * math.cos(th), R * math.sin(th)])

    # ---- quad elements: one per (axial layer j, hoop segment k) --------------
    # winding: (j,k) -> (j,k+1) -> (j+1,k+1) -> (j+1,k)  (consistent quad)
    elements, oris = [], []
    for j in range(NL):
        for k in range(NC):
            k1 = (k + 1) % NC
            n0 = j * NC + k
            n1 = j * NC + k1
            n2 = (j + 1) * NC + k1
            n3 = (j + 1) * NC + k
            elements.append([n0 + 1, n1 + 1, n2 + 1, n3 + 1])

            # per-element frame at the hoop-midpoint angle of this quad
            tm = 2.0 * math.pi * (k + 0.5) / NC
            c, s = math.cos(tm), math.sin(tm)
            oris.append([1.0, 0.0, 0.0,      # e1 = axial
                         0.0, -s, c,         # e2 = hoop tangent
                         0.0, -c, -s])       # e3 = inward normal

    ne = len(elements)
    d = {
        "nodes": nodes,
        "elements": elements,
        "sections": [{"elementSet": "tube", "layup": layup_of_t(t)}],
        "sets": {"element": [{"name": "tube", "labels": list(range(1, ne + 1))}]},
        "materials": [{"name": material["name"], "density": material["density"],
                       "elastic": {"E": material["E"], "G": material["G"], "nu": material["nu"]}}],
        "elementOrientations": oris,
    }
    fn = os.path.join(OUT, "seg_%s_hR%s.yaml" % (tag, hR))
    with open(fn, "w") as f:
        yaml.safe_dump(d, f, default_flow_style=None, sort_keys=False)
    print("wrote %-28s  nodes=%d  quads=%d  t=%.4f  R=%.2f  L=%.2f"
          % (os.path.basename(fn), len(nodes), ne, t, R, L))
    return fn


# layup builders (outer ply listed first; e3 is inward, so "outer" is the -e3 side)
def iso_layup(t):
    return [["iso", t, 0.0]]                       # single 0-degree ply


def aniso_layup(t):
    return [["ani", t / 2.0, 45.0],                # outer half: +45
            ["ani", t / 2.0, -45.0]]               # inner half: -45


if __name__ == "__main__":
    for hR in HR:
        build_segment(ISO, iso_layup, "iso", hR)
        build_segment(ANI, aniso_layup, "aniso", hR)
    print("\nDone. Beam axis = x; end rings at x = 0 and x = %.2f are the two boundary SGs." % L)
