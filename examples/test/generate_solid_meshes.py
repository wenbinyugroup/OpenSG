"""
Generate every FEniCS 2D-solid cross-section YAML used in the MSG thin-walled
benchmark, ready to feed to ``run_solid_timo_from_yaml.py`` (which calls
``opensg.mesh.segment.SolidBounMesh`` + ``opensg.core.solid.compute_timo_boun``).

Four case families are written to ``meshes/`` (no FEniCS needed to generate --
pure numpy):

  1. tube_{iso,aniso}_hR{hr}.yaml      circular tube, R=1 m, h/R in {0.01,0.03,0.06,0.12,0.20}
  2. strip_{iso,aniso}_hW{hr}.yaml     flat strip, W=1 m, h/W in {0.01,0.03,0.06,0.12,0.20}
  3. stripwidth_iso_W{W}_h0.05.yaml    width sweep: h=0.05 m fixed, W in {0.1..2.5}  (h/W 0.5 -> 0.02)
  4. stripthick_iso_h{h}_W1.0.yaml     thickness sweep: W=1 m fixed, h in {0.01..0.5} (h/W 0.01 -> 0.5)

Anisotropic cases are [-45/45] through-thickness (tube: radial) bands with the
fibre baked into the per-element material frame EE1.  Sections are centred at the
origin so the centroid coincides with the centre-referenced shell.

Material frame per element is the 9-component row [e1x,e1y,e1z, e2x,e2y,e2z,
e3x,e3y,e3z]; e3 is the through-thickness normal, e1 the fibre direction.

Order of the resulting Timoshenko 6x6 is [ext, shear2, shear3, twist, bend2, bend3].

Run:  python generate_solid_meshes.py
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "meshes")
os.makedirs(OUT, exist_ok=True)

ISO = dict(E=[70e9, 70e9, 70e9], G=[26.923e9] * 3, nu=[0.3, 0.3, 0.3])
ANI = dict(E=[37e9, 9e9, 9e9], G=[4e9, 4e9, 4e9], nu=[0.3, 0.3, 0.3])


def _write(path, nodes, elems, oris, mat, setname):
    with open(path, "w") as f:
        f.write("nodes:\n")
        for y2, y3 in nodes:
            f.write(f" - [{y2:.10f} {y3:.10f} 0.0]\n")
        f.write("elements:\n")
        for e in elems:
            f.write(f" - [{e[0]} {e[1]} {e[2]} {e[3]}]\n")
        f.write(f"sets:\n  element:\n  - name: {setname}\n    labels: ["
                + ", ".join(str(j + 1) for j in range(len(elems))) + "]\n")
        f.write(f"materials:\n - name: {setname}\n")
        f.write(f"   E: {mat['E']}\n   G: {mat['G']}\n   nu: {mat['nu']}\n   rho: 1800.0\n")
        f.write("elementOrientations:\n")
        for o in oris:
            f.write(" - " + str([round(float(v), 10) for v in o]) + "\n")


def gen_tube(path, R, H, mat, layup, NC=200, NR=16):
    """Annulus, mid-radius R, wall H. layup = [(angle_deg, frac_lo, frac_hi)] inner->outer."""
    ri, ro = R - H / 2, R + H / 2
    rad = np.linspace(ri, ro, NR + 1)
    ang = np.array([2 * np.pi * k / NC for k in range(NC)])
    nid = lambda i, k: i * NC + (k % NC)
    nodes = [(rad[i] * np.cos(ang[k]), rad[i] * np.sin(ang[k]))
             for i in range(NR + 1) for k in range(NC)]
    elems, oris = [], []
    for i in range(NR):
        fr = (i + 0.5) / NR
        thf = np.deg2rad(next(a for a, lo, hi in layup if lo <= fr < hi))
        cf, sf = np.cos(thf), np.sin(thf)
        for k in range(NC):
            a, b = nid(i, k), nid(i, k + 1)
            c, d = nid(i + 1, k + 1), nid(i + 1, k)
            elems.append((a + 1, b + 1, c + 1, d + 1))
            t = ang[k] + np.pi / NC
            ct, st = np.cos(t), np.sin(t)
            oris.append([sf * (-st), sf * ct, cf, cf * (-st), cf * ct, -sf, ct, st, 0.0])
    _write(path, nodes, elems, oris, mat, "tube")


def gen_strip(path, W, H, mat, layup, NC=120, NR=16):
    """Rectangle W x H centred at origin. layup = [(angle_deg, frac_lo, frac_hi)] bottom->top."""
    y2 = np.linspace(-W / 2, W / 2, NC + 1)
    y3 = np.linspace(-H / 2, H / 2, NR + 1)
    nid = lambda i, k: i * (NC + 1) + k
    nodes = [(y2[k], y3[i]) for i in range(NR + 1) for k in range(NC + 1)]
    elems, oris = [], []
    for i in range(NR):
        fr = (i + 0.5) / NR
        thf = np.deg2rad(next(a for a, lo, hi in layup if lo <= fr < hi))
        cf, sf = np.cos(thf), np.sin(thf)
        for k in range(NC):
            a, b = nid(i, k), nid(i, k + 1)
            c, d = nid(i + 1, k + 1), nid(i + 1, k)
            elems.append((a + 1, b + 1, c + 1, d + 1))
            # EE1=fibre=(sf,0,cf), EE2=(cf,0,-sf), EE3=(0,1,0) through-thickness
            oris.append([sf, 0.0, cf, cf, 0.0, -sf, 0.0, 1.0, 0.0])
    _write(path, nodes, elems, oris, mat, "strip")


ISO_LAYUP = [(0.0, 0.0, 1.0)]
ANI_LAYUP = [(-45.0, 0.0, 0.5), (45.0, 0.5, 1.0)]


def main():
    n = 0
    # 1. Tube h/R sweep (R=1), iso + aniso [-45/45]
    R, HR = 1.0, [0.01, 0.03, 0.06, 0.12, 0.20]
    for hr in HR:
        gen_tube(os.path.join(OUT, f"tube_iso_hR{hr}.yaml"), R, hr * R, ISO, ISO_LAYUP); n += 1
        gen_tube(os.path.join(OUT, f"tube_aniso_hR{hr}.yaml"), R, hr * R, ANI, ANI_LAYUP); n += 1
    # 2. Strip h/W sweep (W=1), iso + aniso [-45/45]
    W = 1.0
    for hr in HR:
        gen_strip(os.path.join(OUT, f"strip_iso_hW{hr}.yaml"), W, hr * W, ISO, ISO_LAYUP); n += 1
        gen_strip(os.path.join(OUT, f"strip_aniso_hW{hr}.yaml"), W, hr * W, ANI, ANI_LAYUP); n += 1
    # 3. Strip width sweep: h=0.05 fixed, W varies (h/W 0.5 -> 0.02), isotropic
    Hfix, WIDTHS = 0.05, [0.1, 0.2, 0.35, 0.6, 1.0, 1.6, 2.5]
    for w in WIDTHS:
        gen_strip(os.path.join(OUT, f"stripwidth_iso_W{w}_h0.05.yaml"), w, Hfix, ISO, ISO_LAYUP,
                  NC=160, NR=16); n += 1
    # 4. Strip thickness sweep: W=1 fixed, h varies (h/W 0.01 -> 0.5), isotropic
    Wfix, THICKS = 1.0, [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    for h in THICKS:
        gen_strip(os.path.join(OUT, f"stripthick_iso_h{h}_W1.0.yaml"), Wfix, h, ISO, ISO_LAYUP,
                  NC=160, NR=16); n += 1
    print(f"wrote {n} solid YAML meshes to {OUT}")


if __name__ == "__main__":
    main()
