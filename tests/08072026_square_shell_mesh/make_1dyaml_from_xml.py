"""1-D shell SG yaml from the SAME PreVABS XML, with the SAME orientation.

Reads prevabs/square_tube.xml + prevabs/materials.xml and emits the 1-D (line
element) shell SG on the wall MIDLINE, with:
  * elementOrientations  = [e1 | e2 | e3] per element, e1 = beam axis (z),
    e2 = wall tangent, e3 = e1 x e2 (inward wall normal) -- the shell-yaml
    convention used by opensg_shell / load_ring_ref;
  * the ply angle from the XML layup (-45 deg) carried in sections[].layup,
    so the fibre sits at the same -45 deg from the beam axis as in the 2-D
    solid yaml (there the fibre tilt shows up as e1_z = cos(45) = 0.7071);
  * reference: center  -- the XML baseline is the OUTER profile offset by t/2,
    so the midline used here is that baseline pulled back inward by t/2.

Run (from this folder):  python make_1dyaml_from_xml.py
"""
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

############### User Input #################################
XML = "prevabs/square_tube.xml"
MATXML = "prevabs/materials.xml"
OUT = "square_tube_1Dshell.yaml"
nseg = 10                 # line elements per straight baseline span
############################################################

# ---------------- read the PreVABS XML --------------------------------------
root = ET.parse(XML).getroot()
pts = {}
for p in root.iter("point"):
    v = p.text.split()
    pts[p.get("name")] = np.array([float(v[0]), float(v[1])])
bl = root.find(".//baseline")
seq = [s.strip() for s in bl.find("points").text.split(",")]
closed = seq[0] == seq[-1]
poly = np.array([pts[s] for s in (seq[:-1] if closed else seq)])

layer = root.find(".//layup/layer")
ply_angle = float(layer.text.split(":")[0])
n_plies = int(layer.text.split(":")[1])
lamina = layer.get("lamina")
side = root.find(".//segment/layup").get("direction")

mroot = ET.parse(MATXML).getroot()
lam = [l for l in mroot.iter("lamina") if l.get("name") == lamina][0]
t_ply = float(lam.find("thickness").text)
matname = lam.find("material").text
mat = [m for m in mroot.iter("material") if m.get("name") == matname][0]
el = mat.find("elastic")
g = lambda k: float(el.find(k).text)
E = [g("e1"), g("e2"), g("e3")]
G = [g("g12"), g("g13"), g("g23")]
nu = [g("nu12"), g("nu13"), g("nu23")]
rho = float(mat.find("density").text)
t_w = t_ply * n_plies
print("XML: %d baseline points (closed=%s), layup %s %+.0f deg x%d, t=%.4f, "
      "side=%s" % (len(poly), closed, matname, ply_angle, n_plies, t_w, side))

# ---------------- baseline -> MIDLINE (center reference) --------------------
# the XML baseline is the OUTER profile (midline offset outward by t/2);
# pull it back inward by t/2.  Signed area < 0 => clockwise => inward is the
# LEFT normal of the travel direction.
sa = 0.5*np.sum(poly[:, 0]*np.roll(poly[:, 1], -1)
                - np.roll(poly[:, 0], -1)*poly[:, 1])
ccw = sa > 0
print("baseline signed area = %+.6f (%s)" % (sa, "CCW" if ccw else "CW"))
ctr = poly.mean(0)
mid = np.array([p + (ctr - p)/np.linalg.norm(ctr - p, ord=np.inf)*(t_w/2)
                for p in poly])          # corner-wise inward offset by t/2
if not ccw:
    # PreVABS traverses CW (so that direction="right" lays the ply inward);
    # the shell yamls in this repo are CCW so that e3 = e1 x e2 is the INWARD
    # wall normal (opensg_shell / load_ring_ref convention).  Reverse -- and
    # NEGATE THE PLY ANGLE with it: the fibre is cos(t)*e1 + sin(t)*e2, so
    # reversing the contour reverses e2 and would silently turn a -45 layup
    # into a +45 one, flipping every anisotropic coupling (extension-twist,
    # shear-bending) while leaving the diagonal untouched.
    mid = mid[::-1]
    ply_angle = -ply_angle
    print("reversed baseline to CCW (e3 = inward normal) and negated the ply "
          "angle to %+.1f deg so the fibre direction is unchanged" % ply_angle)

# ---------------- discretise the midline into line elements -----------------
nodes, tans = [], []
npoly = len(mid)
for k in range(npoly):
    p0, p1 = mid[k], mid[(k+1) % npoly]
    tvec = (p1-p0)/np.linalg.norm(p1-p0)
    for i in range(nseg):
        nodes.append(p0 + (p1-p0)*i/nseg)
        tans.append(tvec)
nodes = np.array(nodes); tans = np.array(tans)
m = len(nodes)
cells = np.array([[i, (i+1) % m] for i in range(m)], int)

# ---------------- orientations: e1 = axial z, e2 = tangent, e3 = e1 x e2 ----
ori = []
for k in range(m):
    e1 = np.array([0.0, 0.0, 1.0])
    e2 = np.array([tans[k, 0], tans[k, 1], 0.0])
    e3 = np.cross(e1, e2)
    ori.append(list(e1) + list(e2) + list(e3))
ori = np.array(ori)

# ---------------- write the 1-D shell yaml ----------------------------------
out = ["nodes:"]
for x, y in nodes:
    out.append("- [%.12f %.12f 0.00000000]" % (x, y))
out.append("elements:")
for n1, n2 in cells:
    out.append("- [%d %d]" % (n1+1, n2+1))
out.append("elementOrientations:")
for r in ori:
    out.append("- [" + ", ".join("%.12f" % v for v in r) + "]")
out += ["sections:",
        "- type: shell",
        "  elementSet: layup_0",
        "  layup:"]
for _ in range(n_plies):
    out += ["  - - %s" % matname,
            "    - %.6e" % t_ply,
            "    - %.1f" % ply_angle]
out += ["materials:",
        "- name: %s" % matname,
        "  density: %.1f" % rho,
        "  elastic:",
        "    E: [%.6e, %.6e, %.6e]" % tuple(E),
        "    G: [%.6e, %.6e, %.6e]" % tuple(G),
        "    nu: [%.6f, %.6f, %.6f]" % tuple(nu),
        "sets:",
        "  element:",
        "  - name: layup_0",
        "    labels:"]
out += ["    - %d" % (e+1) for e in range(m)]
out.append("reference: center")
with open(OUT, "w") as f:
    f.write("\n".join(out) + "\n")
print("\nwrote %s: %d nodes, %d line elements, ply %+.0f deg, t=%.4f, ref=center"
      % (OUT, m, len(cells), ply_angle, t_w))

# ---------------- checks -----------------------------------------------------
side_len = np.linalg.norm(mid[1] - mid[0])
print("midline bbox: x [%.4f, %.4f]  y [%.4f, %.4f]  (side %.4f)"
      % (nodes[:, 0].min(), nodes[:, 0].max(),
         nodes[:, 1].min(), nodes[:, 1].max(), side_len))
print("wall area from contour = %.8f  (perimeter %.6f x t)"
      % (side_len*npoly*t_w, side_len*npoly))
dots = np.abs(np.einsum('ij,ij->i', ori[:, 0:3], ori[:, 3:6]))
print("orthonormality: max|e1.e2| = %.2e ; |e3| = %.6f"
      % (dots.max(), np.linalg.norm(ori[0, 6:9])))
print("e3 . outward radial (should be < 0 = inward): %.3f"
      % float(np.dot(ori[0, 6:9][:2], nodes[0] - nodes.mean(0))))

fig, ax = plt.subplots(figsize=(6, 6))
for n1, n2 in cells:
    ax.plot(nodes[[n1, n2], 0], nodes[[n1, n2], 1], "-", color="C0", lw=1.4)
ax.plot(nodes[:, 0], nodes[:, 1], "k.", ms=4)
sc = 0.06
ax.quiver(nodes[:, 0], nodes[:, 1], ori[:, 3], ori[:, 4], color="g",
          scale=1/sc, scale_units="xy", width=0.005, label="e2 (tangent)")
ax.quiver(nodes[:, 0], nodes[:, 1], ori[:, 6], ori[:, 7], color="b",
          scale=1/sc, scale_units="xy", width=0.005, label="e3 (normal)")
ax.set_aspect("equal"); ax.set_xlabel("y2"); ax.set_ylabel("y3")
ax.legend(loc="center", fontsize=9)
fig.tight_layout(); fig.savefig("square_tube_1Dshell.png", dpi=200)
print("wrote square_tube_1Dshell.png")
