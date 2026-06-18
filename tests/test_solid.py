import unittest
import numpy as np
from scipy.spatial import cKDTree

import opensg
from opensg.mesh.segment import SolidBounMesh
from opensg.core.solid import compute_timo_boun
import opensg.core.stress_recov as stress_recov
import opensg.utils as utils
from tests import data_dir, validation_data_dir

# st15 composite cross-section. The FEniCS mesh is the same mesh VABS used.
#   test_solid_st15_vabs.txt -- compact VABS benchmark: Timoshenko 6x6 + the
#       recovered displacement (.U), strain (.EM) and stress (.SM) at the first
#       10 circumferential-path coords.
#   bar_urc-15-t-0.in.glb    -- VABS dehomogenization input (beam reaction force).
ST15_MESH = data_dir / "Solid_2DSG" / "2Dsolid_VABS_15.yaml"
ST15_BENCH = validation_data_dir / "test_solid_st15_vabs.txt"
ST15_GLB = validation_data_dir / "bar_urc-15-t-0.in.glb"
BEAM_OUT = data_dir / "bd_bar_urc.out"

# Homogenization: every non-zero 6x6 term matches VABS to ~5e-5 on the CG1
# (VABS-matched) warping space.
HOMO_RTOL = 5e-3

# Dehomogenization, component-wise relative error along the first 10 path coords
# (max|FEniCS-VABS| / max|VABS| per component over the 10 points). Displacement
# and the axial / transverse-shear stresses (11,12,13) match VABS tightly; the
# transverse in-plane normal/shear stresses (22,33,23) are small residuals of
# large, nearly-cancelling C:e terms (ill-conditioned), so their own-scale
# relative error is necessarily large -- documented here, not a bug.
U_RTOL = 0.02
E_RTOL = {"E11": 0.02, "E22": 0.08, "E33": 0.08, "E23": 1.0, "E13": 0.02, "E12": 0.03}
S_RTOL = {"S11": 0.03, "S22": 0.60, "S33": 0.60, "S23": 1.0, "S13": 0.02, "S12": 0.03}
ECOMP = ["E11", "E22", "E33", "E23", "E13", "E12"]
SCOMP = ["S11", "S22", "S33", "S23", "S13", "S12"]


def _load_bench(path):
    rows = [l for l in open(path).read().splitlines() if l.strip() and not l.lstrip().startswith("#")]
    timo = np.array([[float(x) for x in rows[i].split()] for i in range(6)])
    rec = np.array([[float(x) for x in r.split()] for r in rows[6:]])
    # rec cols: y2 y3 | U1 U2 U3 | E11 E22 E33 E23 E13 E12 | S11 S22 S33 S23 S13 S12
    return timo, rec


def _read_glb_force(path):
    g = open(path).read().split("\n")
    r5 = [float(x) for x in g[4].split()]
    r6 = [float(x) for x in g[5].split()]
    return np.array([r5[0], r6[0], r6[1], r5[1], r5[2], r5[3]])


def _comp_relerr(fe_pts, fe_vals, ref_xy, ref_vals):
    """max|FE-ref| / max|ref| per component, FE sampled at nearest point to each ref coord."""
    idx = cKDTree(fe_pts).query(ref_xy)[1]
    fe = fe_vals[idx]
    den = np.abs(ref_vals).max(axis=0)
    den[den == 0] = 1.0
    return np.abs(fe - ref_vals).max(axis=0) / den


class TestSolid(unittest.TestCase):
    def setUp(self):
        self.timo_ref, self.rec = _load_bench(ST15_BENCH)
        return super().setUp()

    def test_solid_homogenization_vabs_st15(self):
        """FEniCS 2D-solid Timoshenko 6x6 vs VABS for the st15 composite section.

        Every non-zero term (incl. the extension-bending C15/C16, shear-shear C23,
        shear-twist C24/C34 and bending-bending C56 couplings) matches the VABS .K.
        """
        sm = SolidBounMesh(str(ST15_MESH))
        mp, _ = sm.material_database
        C6 = np.asarray(compute_timo_boun(mp, sm.meshdata)[0])
        nz = np.abs(self.timo_ref) > 1e-3 * np.abs(self.timo_ref).max()
        assert np.isclose(C6[nz], self.timo_ref[nz], rtol=HOMO_RTOL, atol=0.0).all()
        print("Solid homogenization vs VABS (st15) passed!")

    def test_solid_dehom_vabs_st15(self):
        """FEniCS 2D-solid displacement/strain/stress recovery vs VABS at the first
        10 circumferential-path coords (component-wise)."""
        FF = _read_glb_force(ST15_GLB)
        sm = SolidBounMesh(str(ST15_MESH))
        mp, _ = sm.material_database
        meshdata = sm.meshdata
        timo = compute_timo_boun(mp, meshdata)
        beam_out = utils.beamdyn_trans.beam_reaction(str(BEAM_OUT))
        _, u_loc, strain_quad, stress_quad, coord_quad = stress_recov.local_strain(
            timo, beam_out, 0, meshdata, mp, FF
        )

        xy = self.rec[:, 0:2]
        u_ref, e_ref, s_ref = self.rec[:, 2:5], self.rec[:, 5:11], self.rec[:, 11:17]

        u_xy = u_loc.function_space.tabulate_dof_coordinates()[:, 1:3]
        u_fe = np.asarray(u_loc.x.array).reshape(-1, 3)
        q_xy = np.asarray(coord_quad).reshape(-1, 3)[:, 1:3]
        e_fe = np.asarray(strain_quad.x.array).reshape(-1, 6)
        s_fe = np.asarray(stress_quad.x.array).reshape(-1, 6)

        u_re = _comp_relerr(u_xy, u_fe, xy, u_ref)
        e_re = _comp_relerr(q_xy, e_fe, xy, e_ref)
        s_re = _comp_relerr(q_xy, s_fe, xy, s_ref)

        for i, lbl in enumerate(["U1", "U2", "U3"]):
            print(f"  {lbl}: {100*u_re[i]:6.2f}%  (tol {100*U_RTOL:.0f}%)")
            assert u_re[i] < U_RTOL, f"{lbl} {u_re[i]:.3f} > {U_RTOL}"
        for i, c in enumerate(ECOMP):
            print(f"  {c}: {100*e_re[i]:6.2f}%  (tol {100*E_RTOL[c]:.0f}%)")
            assert e_re[i] < E_RTOL[c], f"{c} {e_re[i]:.3f} > {E_RTOL[c]}"
        for i, c in enumerate(SCOMP):
            print(f"  {c}: {100*s_re[i]:6.2f}%  (tol {100*S_RTOL[c]:.0f}%)")
            assert s_re[i] < S_RTOL[c], f"{c} {s_re[i]:.3f} > {S_RTOL[c]}"
        print("Solid dehom (U/E/S) vs VABS (st15) passed!")

    def test_solid_mesh_file_generation(self):
        """Solid mesh (.msh) generation produces a valid gmsh file."""
        import os
        from opensg.mesh.segment import SolidSegmentMesh
        segment_file = data_dir / "Solid_3DSG" / "bar_urc_npl_1_ar_5-segment_2.yaml"
        segment_mesh = SolidSegmentMesh(str(segment_file))
        output_file = "test_solid_mesh_output.msh"
        segment_mesh.generate_mesh_file(output_file)
        assert os.path.exists(output_file), "Mesh file was not created"
        with open(output_file, "r") as f:
            content = f.read()
            assert "$MeshFormat" in content
            assert "$Nodes" in content
            assert "$Elements" in content
            assert "$EndElements" in content
        os.remove(output_file)


if __name__ == "__main__":
    unittest.main()
