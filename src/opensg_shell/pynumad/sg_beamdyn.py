"""OpenSG <-> BeamDyn coupling: .K reading, frame transforms, BeamDyn input writers,
the beamdyn_driver runner, and per-station .ff extraction for the dehomogenization.

Frames.  Cross-section matrices/loads are in the VABS frame (1 = axial x1); BeamDyn uses
the IEC blade frame (z = span, x = flap).  The map is the beam-axis swap
B = [[0,0,1],[0,-1,0],[1,0,0]]: vectors v_bd = B^T v_vabs; 6x6 blocks transform by the
similarity T = B^{-1} applied block-wise.  BeamDyn's RD* outputs are Wiener-Milenkovic
parameters c = 4 tan(phi/4) n (NOT small angles): the direction-cosine matrix is the
exact Rodrigues tensor (Wang et al., SDM 2013, Eq. 12), and a finite rotation remaps
between frames by the similarity C_vabs = B C_bd B^T -- never a component swap.

The per-station .ff file (glb-type recovery input, VABS frame; values first, inline
labels, blocks separated by blank lines):
    # OpenSG station FF (BeamDyn -> VABS frame)  eta = 0.020000

    <u1 u2 u3>        # u1 u2 u3 [m];
    <th1 th2 th3>     # th1 th2 th3 (Wiener-Milenkovic);

    <C row 1>         # C 3x3;
    <C row 2>
    <C row 3>

    <F1 F2 F3>        # F1 F2 F3 [N]
    <M1 M2 M3>        # M1 M2 M3 [N-m]
C is the deformed-triad rotation (C_bB^T storage, as VABS .glb).  Forces/moments are
the section-LOCAL ('L') BeamDyn resultants (VABS recovery is in the section frame;
the DCM rotates the recovered field back to global; F1 = axial, M1 = torsion).
"""
import os
import subprocess

import numpy as np

B_MAP = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])


# ----------------------------------------------------------------- .K reading + transforms
def read_k_file(path):
    """Read the Timoshenko stiffness and mass 6x6 from a VABS .K-layout file.

    Accepts both the sg_props.write_vabs_k output and a native VABS .K (and the legacy
    OpenSG "Stiffness :" block as a fallback).

    This is the historical msg-shell import path and stays exactly that; the parser
    itself was promoted to opensg_solid.io.k_file (read_beam_k), the ONE .K reader
    both engines and the tests use -- that module also reads the SwiftComp 3-D solid
    .sc.k and carries the compare_to_K verifier.

    In:
        path: str, .K (or .out) file.
    Out:
        (K, M): (6,6) Timoshenko stiffness, (6,6) mass matrix (VABS order).
    """
    from opensg_solid.io.k_file import read_beam_k
    return read_beam_k(path)


def _trsf_sixbysix(Mtx, T):
    """Block-wise similarity transform of a 6x6 (Voigt reference-frame change).

    In:
        Mtx: (6,6); T: (3,3) transformation.
    Out:
        (6,6) transformed matrix.
    """
    out = np.zeros((6, 6))
    for a in range(2):
        for b in range(2):
            out[3 * a:3 * a + 3, 3 * b:3 * b + 3] = \
                T.T @ Mtx[3 * a:3 * a + 3, 3 * b:3 * b + 3] @ T
    return out


def vabs_to_beamdyn(K, M):
    """VABS-frame stiffness/mass -> BeamDyn blade frame (beam-axis swap).

    In:
        K, M: (6,6) VABS-frame matrices (or (n,6,6) stacks).
    Out:
        (K_bd, M_bd) with the same shape.
    """
    T = np.linalg.inv(B_MAP)
    K = np.asarray(K, float); M = np.asarray(M, float)
    if K.ndim == 2:
        return _trsf_sixbysix(K, T), _trsf_sixbysix(M, T)
    Kb = np.stack([_trsf_sixbysix(K[i], T) for i in range(K.shape[0])])
    Mb = np.stack([_trsf_sixbysix(M[i], T) for i in range(M.shape[0])])
    return Kb, Mb


# ----------------------------------------------------------------- BeamDyn input writers
def write_bd_props(path, etas, K_bd, M_bd, mu=None):
    """Write the BeamDyn blade (distributed properties) input file.

    In:
        path: str, output .inp.
        etas: (n,) non-dimensional span stations.
        K_bd, M_bd: (n,6,6) BeamDyn-frame stiffness / mass per station.
        mu: (6,) damping coefficients (default 1e-3 each, damp_type 1).
    Out:
        str path.
    """
    mu = [1e-3] * 6 if mu is None else list(mu)
    n = len(etas)
    with open(path, "w") as f:
        f.write(" ------- BEAMDYN V1.00.* INDIVIDUAL BLADE INPUT FILE --------------------------\n")
        f.write(" OpenSG msg-shell blade sectional properties\n")
        f.write(" ---------------------- BLADE PARAMETERS --------------------------------------\n")
        f.write("%u   station_total    - Number of blade input stations (-)\n" % n)
        f.write(" 1   damp_type        - Damping type: 0: no damping; 1: damped\n")
        f.write("  ---------------------- DAMPING COEFFICIENT------------------------------------\n")
        f.write("   mu1        mu2        mu3        mu4        mu5        mu6\n")
        f.write("   (-)        (-)        (-)        (-)        (-)        (-)\n")
        f.write("\t %.5e \t %.5e \t %.5e \t %.5e \t %.5e \t %.5e\n" % tuple(mu))
        f.write(" ---------------------- DISTRIBUTED PROPERTIES---------------------------------\n")
        for i in range(n):
            f.write("\t %.6f \n" % etas[i])
            for j in range(6):
                f.write("\t %.16e \t %.16e \t %.16e \t %.16e \t %.16e \t %.16e\n"
                        % tuple(K_bd[i, j]))
            f.write("\n")
            for j in range(6):
                f.write("\t %.16e \t %.16e \t %.16e \t %.16e \t %.16e \t %.16e\n"
                        % tuple(M_bd[i, j]))
            f.write("\n")
    return path


_BLDND_CHANNELS = ["FxL", "FyL", "FzL", "Fxr", "Fyr", "Fzr",
                   "MxL", "MyL", "MzL", "TDxr", "TDyr", "TDzr",
                   "RDxr", "RDyr", "RDzr"]


def write_bd_primary(path, blade_length, props_file, title="OpenSG blade",
                     kp_spacing=2.0, order_elem=10, refine=1):
    """Write the BeamDyn primary input: straight untwisted reference line, TRAPEZOIDAL
    quadrature so the output nodes coincide with the blade property stations.

    The section 6x6 is in the local frame with no prebend context, so the reference
    line adds none (kp_xr = kp_yr = twist = 0).  quadrature=2 places the
    quadrature/output nodes on the property-station grid (Gaussian only outputs at
    the order_elem+1 GLL nodes); ``refine`` inserts refine-1 interpolated quadrature
    points between consecutive stations -- needed when the station set is sparse
    (e.g. the 16 windIO airfoil stations: refine=1 does not converge statically) --
    and every station remains an output node at index i*refine + 1.

    In:
        path: str, output .inp.
        blade_length: float [m].
        props_file: str, blade-properties filename (as referenced from the primary).
        title: str, line-2 description.
        kp_spacing: float [m], target key-point spacing along the span.
        order_elem: int, element interpolation order.
        refine: int, trapezoidal refinement factor (match extract_ff's ``refine``).
    Out:
        str path.
    """
    nkp = max(3, int(round(blade_length / kp_spacing)))
    zs = np.linspace(0.0, blade_length, nkp)
    with open(path, "w") as f:
        f.write("--------- BEAMDYN with OpenFAST INPUT FILE -------------------------------------------\n")
        f.write("%s\n" % title)
        f.write("---------------------- SIMULATION CONTROL --------------------------------------\n")
        f.write("True          Echo             - Echo input data to \"<RootName>.ech\"? (flag)\n")
        f.write("False         QuasiStaticInit  - Use quasi-static pre-conditioning with centripetal accelerations in initialization? (flag) [dynamic solve only]\n")
        f.write("          0   rhoinf           - Numerical damping parameter for generalized-alpha integrator\n")
        f.write("          2   quadrature       - Quadrature method: 1=Gaussian; 2=Trapezoidal (switch)\n")
        f.write("          %d   refine           - Refinement factor for trapezoidal quadrature (-) [DEFAULT = 1; used only when quadrature=2]\n" % refine)
        f.write("\"DEFAULT\"     n_fact           - Factorization frequency for the Jacobian in N-R iteration(-) [DEFAULT = 5]\n")
        f.write("\"DEFAULT\"     DTBeam           - Time step size (s)\n")
        f.write("50     load_retries     - Number of factored load retries before quitting the simulation [DEFAULT = 20]\n")
        f.write("\"DEFAULT\"     NRMax            - Max number of iterations in Newton-Raphson algorithm (-) [DEFAULT = 10]\n")
        f.write("\"DEFAULT\"     stop_tol         - Tolerance for stopping criterion (-) [DEFAULT = 1E-5]\n")
        f.write("\"DEFAULT\"     tngt_stf_fd      - Use finite differenced tangent stiffness matrix? (flag)\n")
        f.write("\"DEFAULT\"     tngt_stf_comp    - Compare analytical finite differenced tangent stiffness matrix? (flag)\n")
        f.write("\"DEFAULT\"     tngt_stf_pert    - Perturbation size for finite differencing (-) [DEFAULT = 1E-6]\n")
        f.write("\"DEFAULT\"     tngt_stf_difftol - Maximum allowable relative difference between analytical and fd tangent stiffness (-); [DEFAULT = 0.1]\n")
        f.write("True          RotStates        - Orient states in the rotating frame during linearization? (flag) [used only when linearizing] \n")
        f.write("---------------------- GEOMETRY PARAMETER --------------------------------------\n")
        f.write("          1   member_total    - Total number of members (-)\n")
        f.write("%11d   kp_total        - Total number of key points (-) [must be at least 3]\n" % nkp)
        f.write("     1     %d                 - Member number; Number of key points in this member\n" % nkp)
        f.write("\t\t kp_xr \t\t\t kp_yr \t\t\t kp_zr \t\t initial_twist\n")
        f.write("\t\t  (m)  \t\t\t  (m)  \t\t\t  (m)  \t\t   (deg)\n")
        for z in zs:
            f.write("\t   %.5e \t   %.5e \t   %.5e \t   %.5e \n" % (0.0, 0.0, z, 0.0))
        f.write("---------------------- MESH PARAMETER ------------------------------------------\n")
        f.write("          %d   order_elem     - Order of interpolation (basis) function (-)\n" % order_elem)
        f.write("---------------------- MATERIAL PARAMETER --------------------------------------\n")
        f.write("\"%s\"    BldFile - Name of file containing properties for blade (quoted string)\n" % props_file)
        f.write("---------------------- PITCH ACTUATOR PARAMETERS -------------------------------\n")
        f.write("False         UsePitchAct - Whether a pitch actuator should be used (flag)\n")
        f.write("          0   PitchJ      - Pitch actuator inertia (kg-m^2) [used only when UsePitchAct is true]\n")
        f.write("          0   PitchK      - Pitch actuator stiffness (kg-m^2/s^2) [used only when UsePitchAct is true]\n")
        f.write("          0   PitchC      - Pitch actuator damping (kg-m^2/s) [used only when UsePitchAct is true]\n")
        f.write("---------------------- OUTPUTS -------------------------------------------------\n")
        f.write("True          SumPrint       - Print summary data to \"<RootName>.sum\" (flag)\n")
        f.write("\"ES16.8E2\"    OutFmt          - Format used for text tabular output, excluding the time channel.\n")
        f.write("          1   NNodeOuts      - Number of nodes to output to file [0 - 9] (-)\n")
        f.write("          1   OutNd          - Nodes whose values will be output  (-)\n")
        f.write("          OutList        - The next line(s) contains a list of output parameters. See OutListParameters.xlsx for a listing of available output channels, (-) \n")
        f.write("\"TipTDxr, TipTDyr, TipTDzr\"  \n")
        f.write("\"TipRDxr, TipRDyr, TipRDzr\" \n")
        f.write("\"RootFxr, RootFyr, RootFzr\"  \n")
        f.write("\"RootMxr, RootMyr, RootMzr\"  \n")
        f.write("END of input file (the word \"END\" must appear in the first 3 columns of this last OutList line)\n")
        f.write("---------------------- NODE OUTPUTS --------------------------------------------\n")
        f.write("         99   BldNd_BlOutNd   - Blade nodes on each blade (currently unused)\n")
        f.write("              OutList     - The next line(s) contains a list of output parameters.  See OutListParameters.xlsx, BeamDyn_Nodes tab for a listing of available output channels, (-)\n")
        for ch in _BLDND_CHANNELS:
            f.write("\"%s\"\n" % ch)
        f.write("END of input file (the word \"END\" must appear in the first 3 columns of this last OutList line)\n")
        f.write("---------------------------------------------------------------------------------------\n")
    return path


def write_bd_driver(path, primary_file, etas, loads, title="OpenSG blade static analysis"):
    """Write the BeamDyn driver input: static solve, gravity off, per-station nodal loads.

    In:
        path: str, output .inp.
        primary_file: str, primary-input filename (as referenced from the driver).
        etas: (n,) non-dimensional stations of the point loads.
        loads: (n,6) [Fx Fy Fz Mx My Mz] in the BeamDyn blade frame (x = flap,
            z = span; a flapwise surface traction gives Fx and a torsion Mz).
        title: str, line-2 description.
    Out:
        str path.
    """
    n = len(etas)
    with open(path, "w") as f:
        f.write("------- BEAMDYN Driver with OpenFAST INPUT FILE --------------------------------\n")
        f.write("%s\n" % title)
        f.write("---------------------- SIMULATION CONTROL --------------------------------------\n")
        f.write("False         DynamicSolve  - Dynamic solve (false for static solve) (-)\n")
        f.write("          0   t_initial     - Starting time of simulation (s) [used only when DynamicSolve=TRUE]\n")
        f.write("         30   t_final       - Ending time of simulation   (s) [used only when DynamicSolve=TRUE]\n")
        f.write("      0.0002   dt            - Time increment size         (s) [used only when DynamicSolve=TRUE]\n")
        f.write("---------------------- GRAVITY PARAMETER --------------------------------------\n")
        f.write("          0   Gx            - Component of gravity vector along X direction (m/s^2)\n")
        f.write("          0   Gy            - Component of gravity vector along Y direction (m/s^2)\n")
        f.write("          0   Gz            - Component of gravity vector along Z direction (m/s^2)\n")
        f.write("---------------------- FRAME PARAMETER --------------------------------------\n")
        f.write("          0   GlbPos(1)     - Component of position vector of the reference blade frame along X direction (m)\n")
        f.write("          0   GlbPos(2)     - Component of position vector of the reference blade frame along Y direction (m)\n")
        f.write("          0   GlbPos(3)     - Component of position vector of the reference blade frame along Z direction (m)\n")
        f.write("---The following 3 by 3 matrix is the direction cosine matirx ,GlbDCM(3,3),\n")
        f.write("---relates global frame to the initial blade root frame\n")
        f.write("1.0000000E+00  0.0000000E+00  0.0000000E+00  \n")
        f.write("0.0000000E+00  1.0000000E+00  0.0000000E+00  \n")
        f.write("0.0000000E+00  0.0000000E+00  1.0000000E+00  \n")
        f.write("T             GlbRotBladeT0 - Reference orientation for BeamDyn calculations is aligned with initial blade root?\n")
        f.write("---------------------- ROOT VELOCITY PARAMETER ----------------------------------\n")
        f.write("          0   RootVel(4)    - Component of angular velocity vector of the beam root about X axis (rad/s)\n")
        f.write("          0   RootVel(5)    - Component of angular velocity vector of the beam root about Y axis (rad/s)\n")
        f.write("          0   RootVel(6)    - Component of angular velocity vector of the beam root about Z axis (rad/s)\n")
        f.write("---------------------- APPLIED FORCE ----------------------------------\n")
        for k, lab in enumerate(["force vector along X direction (N/m)",
                                 "force vector along Y direction (N/m)",
                                 "force vector along Z direction (N/m)",
                                 "moment vector along X direction (N-m/m)",
                                 "moment vector along Y direction (N-m/m)",
                                 "moment vector along Z direction (N-m/m)"]):
            f.write("          0   DistrLoad(%d)  - Component of distributed %s\n" % (k + 1, lab))
        for k, lab in enumerate(["force vector at blade tip along X direction (N)",
                                 "force vector at blade tip along Y direction (N)",
                                 "force vector at blade tip along Z direction (N)",
                                 "moment vector at blade tip along X direction (N-m)",
                                 "moment vector at blade tip along Y direction (N-m)",
                                 "moment vector at blade tip along Z direction (N-m)"]):
            f.write("          0   TipLoad(%d)    - Component of concentrated %s\n" % (k + 1, lab))
        f.write("         %d   NumPointLoads - Number of point loads along blade\n" % n)
        f.write("Non-dim blade-span eta   Fx          Fy            Fz           Mx           My           Mz\n")
        f.write("(-)                      (N)         (N)           (N)          (N-m)        (N-m)        (N-m)\n")
        for i in range(n):
            f.write("%.4f %.10g %.10g %.10g %.10g %.10g %.10g\n"
                    % (etas[i], loads[i][0], loads[i][1], loads[i][2],
                       loads[i][3], loads[i][4], loads[i][5]))
        f.write("---------------------- PRIMARY INPUT FILE --------------------------------------\n")
        f.write("\"%s\"    InputFile - Name of the primary BeamDyn input file\n" % primary_file)
        f.write("----- Output Settings -------------------------------------------------------------------\n")
        f.write("          0   WrVTK         - VTK visualization data output: (switch) {0=none; 1=init; 2=animation}\n")
        f.write("         15   VTK_fps       - Frame rate for VTK output (frames per second) {will use closest integer multiple of DT} [used only if WrVTK=2]\n")
    return path


def station_loads(blade, etas, p_traction, mesh_size=0.01, css=None):
    """Per-station nodal loads from a uniform flapwise surface traction, about x1.

    Per station i with tributary span bin ds_i (ends carry a half bin):
        Fx_i = p * P_i * ds_i             (flapwise force; P = OML perimeter)
        Mz_i = p * int(x2 ds)_i * ds_i    (torsion about x1; x2 from the reference axis)

    In:
        blade: WindIOBlade | WindIOBladeV1.
        etas: (n,) non-dimensional stations.
        p_traction: float [Pa], uniform flapwise OML traction.
        mesh_size: float, cross-section resolution (only the OML contour is used).
        css: list of cs dicts | None, pre-built cross-sections to reuse.
    Out:
        (n,6) float loads [Fx 0 0 0 0 Mz] in the BeamDyn blade frame.
    """
    from .sg_mesh import build_cross_section, oml_load_integrals
    from .sg_pynumad import blade_length

    etas = np.asarray(etas, float)
    L = blade_length(blade)
    b = np.concatenate([[etas[0]], 0.5 * (etas[:-1] + etas[1:]), [etas[-1]]])
    ds_bin = np.diff(b) * L
    loads = np.zeros((len(etas), 6))
    for i, e in enumerate(etas):
        cs = css[i] if css is not None else build_cross_section(blade, float(e), mesh_size)
        P, Sx2ds = oml_load_integrals(cs)
        loads[i, 0] = p_traction * P * ds_bin[i]
        loads[i, 5] = p_traction * Sx2ds * ds_bin[i]
    return loads


# ----------------------------------------------------------------- run + output reading
def run_beamdyn(driver_inp, exe="beamdyn_driver", cwd=None, ld_library_path=None):
    """Run beamdyn_driver on a driver input and return the .out path.

    In:
        driver_inp: str, driver input file.
        exe: str, beamdyn_driver executable (name on PATH or full path).
        cwd: str | None, working directory (default: the driver's directory).
        ld_library_path: str | None, prepended to LD_LIBRARY_PATH (conda builds need
            the conda lib directory).
    Out:
        str, path to <driver base>.out.
    """
    cwd = cwd or (os.path.dirname(os.path.abspath(driver_inp)) or ".")
    env = os.environ.copy()
    if ld_library_path:
        env["LD_LIBRARY_PATH"] = ld_library_path + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    r = subprocess.run([exe, os.path.basename(driver_inp)], cwd=cwd, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = os.path.join(cwd, os.path.splitext(os.path.basename(driver_inp))[0] + ".out")
    if r.returncode != 0 or not os.path.exists(out):
        tail = "\n".join(r.stdout.splitlines()[-25:])
        raise RuntimeError("beamdyn_driver failed (%d):\n%s" % (r.returncode, tail))
    return out


def read_bd_out(path):
    """Read a BeamDyn .out table and return the converged (last) row by channel.

    In:
        path: str, BeamDyn .out.
    Out:
        (d, hdr): dict {channel: value} of the last row; hdr list of channel names.
    """
    lines = [ln for ln in open(path).read().splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if ln.strip().startswith("Time"):
            hdr = ln.split()
            data = np.array([r.split() for r in lines[i + 2:]], float)
            row = data[-1]
            return {h: row[j] for j, h in enumerate(hdr)}, hdr
    raise ValueError("no output header in %s" % path)


def _n_nodes(hdr):
    return sum(1 for h in hdr if h.endswith("_TDxr") and h.startswith("N"))


# ----------------------------------------------------------------- rotations (WM <-> DCM)
def wm_to_dcm(c):
    """Wiener-Milenkovic parameters -> direction-cosine matrix (Wang SDM-2013 Eq. 12).

    In:
        c: (3,) WM parameters (c = 4 tan(phi/4) n).
    Out:
        (3,3) rotation tensor.
    """
    c1, c2, c3 = c
    c0 = 2.0 - (c1 * c1 + c2 * c2 + c3 * c3) / 8.0
    d = (4.0 - c0) ** 2
    return np.array([
        [c0 * c0 + c1 * c1 - c2 * c2 - c3 * c3, 2 * (c1 * c2 - c0 * c3), 2 * (c1 * c3 + c0 * c2)],
        [2 * (c1 * c2 + c0 * c3), c0 * c0 - c1 * c1 + c2 * c2 - c3 * c3, 2 * (c2 * c3 - c0 * c1)],
        [2 * (c1 * c3 - c0 * c2), 2 * (c2 * c3 + c0 * c1), c0 * c0 - c1 * c1 - c2 * c2 + c3 * c3],
    ]) / d


def dcm_to_wm(C):
    """Direction-cosine matrix -> Wiener-Milenkovic parameters (via the unit quaternion).

    q = (q0, qv) from Shepperd's method with q0 >= 0 (|phi| <= pi), then
    c = 4 qv / (1 + q0) = 4 tan(phi/4) n -- the exact inverse of wm_to_dcm.

    In:
        C: (3,3) rotation tensor.
    Out:
        (3,) WM parameters.
    """
    C = np.asarray(C, float)
    tr = np.trace(C)
    q = np.zeros(4)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q[0] = 0.25 * s
        q[1] = (C[2, 1] - C[1, 2]) / s
        q[2] = (C[0, 2] - C[2, 0]) / s
        q[3] = (C[1, 0] - C[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(C)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(C[i, i] - C[j, j] - C[k, k] + 1.0, 0.0)) * 2.0
        qv = np.zeros(3)
        qv[i] = 0.25 * s
        qv[j] = (C[j, i] + C[i, j]) / s
        qv[k] = (C[k, i] + C[i, k]) / s
        q[0] = (C[k, j] - C[j, k]) / s
        q[1:] = qv
    if q[0] < 0.0:
        q = -q
    return 4.0 * q[1:] / (1.0 + q[0])


# ----------------------------------------------------------------- .ff extraction
def _station_state(d, k):
    """Raw BeamDyn-frame state at output node k (1-based).

    In:
        d: dict from read_bd_out; k: int node.
    Out:
        (TD, RD, FF, MM): displacement ('r' frame), WM rotation ('r'), force and
        moment (section-LOCAL 'L'; 'r' fallback when 'L' channels are absent).
    """
    p = "N%03d_" % k

    def g(name, alt=None):
        if p + name in d:
            return d[p + name]
        return d[p + alt] if alt else 0.0
    TD = np.array([g("TDxr"), g("TDyr"), g("TDzr")])
    RD = np.array([g("RDxr"), g("RDyr"), g("RDzr")])
    FF = np.array([g("FxL", "Fxr"), g("FyL", "Fyr"), g("FzL", "Fzr")])
    MM = np.array([g("MxL", "Mxr"), g("MyL", "Myr"), g("MzL", "Mzr")])
    return TD, RD, FF, MM


def _to_vabs(TD, RD, FF, MM, transpose=True):
    """BeamDyn-frame state -> VABS frame.

    Vectors map by B; the finite rotation by the similarity C = B C_bd B^T (transposed
    by default: VABS stores C_bB = (C_Bb)^T, validated against the BAR-URC .glb set).

    In:
        TD, RD, FF, MM: (3,) BeamDyn-frame arrays; transpose: bool.
    Out:
        (u, C, F, M): displacement, DCM, force, moment in the VABS frame.
    """
    u = B_MAP @ TD
    F = B_MAP @ FF
    M = B_MAP @ MM
    C = B_MAP @ wm_to_dcm(RD) @ B_MAP.T
    if transpose:
        C = C.T
    return u, C, F, M


def write_ff(path, eta, u, th, C, F, M):
    """Write one per-station .ff recovery-input file (glb-type, VABS frame): values
    first with inline labels, blocks separated by blank lines (see module header).

    In:
        path: str; eta: float station; u (3,), th (3,) WM params, C (3,3), F (3,),
        M (3,) -- all VABS frame (F1 = axial, M1 = torsion).
    Out:
        str path.
    """
    with open(path, "w") as f:
        f.write("# OpenSG station FF (BeamDyn -> VABS frame)  eta = %.6f\n\n" % eta)
        f.write("%.9g %.9g %.9g # u1 u2 u3 [m];\n" % tuple(u))
        f.write("%.9g %.9g %.9g # th1 th2 th3 (Wiener-Milenkovic);\n\n" % tuple(th))
        f.write("%.16g %.16g %.16g # C 3x3;\n" % tuple(C[0]))
        f.write("%.16g %.16g %.16g\n" % tuple(C[1]))
        f.write("%.16g %.16g %.16g\n\n" % tuple(C[2]))
        f.write("%.9g %.9g %.9g # F1 F2 F3 [N]\n" % tuple(F))
        f.write("%.9g %.9g %.9g # M1 M2 M3 [N-m]\n" % tuple(M))
    return path


def _ff_floats(line):
    """Leading float tokens of a data line (stops at the first non-numeric token, so
    inline label tails parse with or without a '#')."""
    vals = []
    for tok in line.split("#")[0].split():
        try:
            vals.append(float(tok))
        except ValueError:
            break
    return np.array(vals)


def read_ff(path):
    """Read a per-station .ff file (accepts the labeled layout of write_ff and the
    older leading-# header variant).

    In:
        path: str, .ff file from write_ff.
    Out:
        dict: "eta" float, "u" (3,), "th" (3,), "C" (3,3), "F" (3,), "M" (3,),
        "FF" (6,) [F1 F2 F3 M1 M2 M3] (the beam_force_vabs vector for the dehom).
    """
    eta = np.nan
    rows = []
    for ln in open(path).read().splitlines():
        if not ln.strip():
            continue
        if ln.lstrip().startswith("#"):
            if "eta =" in ln:
                eta = float(ln.split("eta =")[1].split()[0])
            continue
        v = _ff_floats(ln)
        if len(v):
            rows.append(v)
    u, th = rows[0], rows[1]
    C = np.vstack(rows[2:5])
    F, M = rows[5], rows[6]
    return dict(eta=eta, u=u, th=th, C=C, F=F, M=M, FF=np.r_[F, M])


def extract_ff(bd_out, etas, out_dir, prefix="station", table=None, refine=1):
    """Extract per-station .ff files (u, theta, DCM, F, M in the VABS frame) from a
    BeamDyn .out (trapezoidal quadrature: output nodes on the refined station grid).

    In:
        bd_out: str, BeamDyn driver .out.
        etas: (n,) station positions (must match the props file's station_total).
        out_dir: str, output directory for <prefix>_rXXXX.ff.
        prefix: str, file-name prefix; tag = round(eta * 1000).
        table: str | None, also write a combined table (# eta F1 F2 F3 M1 M2 M3).
        refine: int, the primary input's trapezoidal refinement factor; station i is
            output node i*refine + 1 of the (n-1)*refine + 1 quadrature nodes.
    Out:
        (files, tab): list of .ff paths; (n,7) array [eta F1 F2 F3 M1 M2 M3].
    """
    d, hdr = read_bd_out(bd_out)
    N = _n_nodes(hdr)
    S = len(etas)
    if N != (S - 1) * refine + 1:
        raise ValueError("BeamDyn wrote %d output nodes; %d stations at refine=%d "
                         "expect %d -- pass the refine used in write_bd_primary"
                         % (N, S, refine, (S - 1) * refine + 1))
    os.makedirs(out_dir, exist_ok=True)
    files = []; tab = np.zeros((S, 7))
    for i in range(S):
        k = i * refine + 1                                # station i -> refined node k
        TD, RD, FF, MM = _station_state(d, k)
        u, C, F, M = _to_vabs(TD, RD, FF, MM)
        th = dcm_to_wm(B_MAP @ wm_to_dcm(RD) @ B_MAP.T)   # WM params of the VABS-frame rotation
        eta = float(etas[i])
        p = os.path.join(out_dir, "%s_r%04d.ff" % (prefix, round(eta * 1000)))
        write_ff(p, eta, u, th, C, F, M)
        files.append(p)
        tab[i] = [eta, F[0], F[1], F[2], M[0], M[1], M[2]]
    if table:
        np.savetxt(table, tab, fmt="%.8e",
                   header="eta F1 F2 F3 M1 M2 M3 (VABS order, section-local resultants)")
    return files, tab
