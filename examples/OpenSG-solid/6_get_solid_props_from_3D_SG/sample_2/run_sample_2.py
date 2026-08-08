"""msg-solid 3-D SG solid properties: TPMS Sample_2 (192k tets), periodic in
all 3 directions, aluminum E = 69 GPa nu = 0.3 (matched to the SwiftComp .K).
Writes the timed .out (per-cell), the mesh png (solver default) and the
comparison .dat vs SwiftComp.  Run:  python run_sample_2.py"""
import numpy as np
import jax.numpy as jnp

from opensg_solid.sg_homo import plate_homo_2d

S = "2"
SC = ("../../../../tests/08072026_square_shell_mesh/TPMS_SC_files/"
      "Sample_%s.sc" % S)
E, nu = 69.0e9, 0.30
G = E/(2*(1+nu))

r = plate_homo_2d(SC, material_param=jnp.array([(E, E, E, G, G, G,
                                                 nu, nu, nu)]),
                  angles=jnp.array([0.0]), n_model=3, workdir=".", plot=True,
                  boundary="periodic")   # SwiftComp .K digit-parity mode
C = np.asarray(r["C_eff"])
C = 0.5*(C + C.T)
om = float(r["omega"])
Cc = C*om                                    # per unit cell (SwiftComp norm)

ln = open(SC + ".k").read().splitlines()
i0 = next(i for i, l in enumerate(ln) if "Effective Stiffness" in l) + 2
K = np.array([[float(v) for v in ln[i0+i].split()]
              for i in range(6)])*1e6        # MPa -> Pa

with open("Sample_%s_msg_solid.out" % S, "w") as f:
    f.write("# TPMS Sample_%s -- msg-solid 3-D SG, periodic all 3 dirs,"
            " alu E=69 GPa nu=0.3\n# per-UNIT-CELL C (Pa) = C_eff x omega;"
            " omega %.6g; solve time: %.2f s\n# ---- C (6x6, Pa) ----\n"
            % (S, om, r["solve_time"]))
    for i in range(6):
        f.write(" ".join("%16.8e" % Cc[i, j] for j in range(6)) + "\n")

TERMS = [("C11", 0, 0), ("C12", 0, 1), ("C13", 0, 2), ("C22", 1, 1),
         ("C23", 1, 2), ("C33", 2, 2), ("C44", 3, 3), ("C55", 4, 4),
         ("C66", 5, 5)]
lines = ["# TPMS Sample_%s: msg-solid vs SwiftComp .K (per-cell, MPa);"
         " relative density %.4f; solve %.1f s" % (S, om, r["solve_time"]),
         "# %-5s %12s %12s %9s" % ("term", "msg_solid", "SwiftComp", "err%")]
for nm, i, j in TERMS:
    lines.append("  %-5s %12.3f %12.3f %+9.3f"
                 % (nm, Cc[i, j]*1e-6, K[i, j]*1e-6,
                    100*(Cc[i, j]-K[i, j])/K[i, j]))
open("Sample_%s_compare.dat" % S, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

