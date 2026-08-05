"""yu62_config.py -- the Yu-2003 SECTION 6.2 dynamic problem (their Fig. 27).

Problem statement (C&S 81 (2003) 439-454, sec. 6.2, Eq. 69): a THICK square
composite plate, width w = 0.04 m, thickness h = 0.01 m (w/h = 4), clamped
along edges BC and CD and free along the other two; a concentrated mass
M = 50 kg attached at the free corner A; two-ply layup [90, 0] (bottom -> top,
the 0 direction parallel to x1); material (SI):

    E1 = 172.4 GPa   E2 = E3 = 6.9 GPa
    G12 = G13 = 3.45 GPa   G23 = 1.38 GPa
    nu12 = nu13 = nu23 = 0.25   rho = 1600 kg/m^3

Load: concentrated transverse force at A, a triangular impulse linearly rising
to 10 kN at t = 0.001 s and back to zero at t = 0.002 s; afterwards FREE
VIBRATION (Yu reports period T = 0.0332 s).  Time step dt = 1e-4 s.  The 3-D
stress fields are recovered through the thickness at t = 0.0096 s at points
M (w/2, w/2, the plate centre) and Q (h, w - h).  Yu couples VAPAS to DYMORE;
here Abaqus/Standard implicit dynamics plays DYMORE and OpenSG-RM recovers.

Corner/edge layout (their Fig. 27): A = (0, 0), B = (w, 0), C = (w, w),
D = (0, w); clamped BC = the x = w edge and CD = the y = w edge.

Module variables (shared by the deck generator and the post-processor)
----------------------------------------------------------------------
MATERIAL_DB   {"yu62": {...}} the SI ply above (rho used by the solid deck)
W, H          plate width 0.04 [m] and thickness 0.01 [m]
THK, ANG,     the layup: two 0.005 m plies, [90, 0] bottom -> top
MATS
PMASS, F0     corner mass 50 [kg]; impulse peak 10e3 [N] (applied -z)
T_RISE,       impulse knots 0.001 / 0.002 [s]; fixed time step 1e-4 [s];
T_END, DT,    total time 0.05 [s]; recovery instant 0.0096 [s]
T_TOTAL,
T_REC
NX            shell mesh: NX x NX S4 elements (24 ~ Yu's 12x12 quadratic)
NZ_PLY        solid mesh: elements through EACH ply (4 -> 8 total)
XM, XQ        the recovery points M = (W/2, W/2) and Q = (H, W - H) [m];
              both land EXACTLY on mesh nodes with NX = 24 (dx = W/24:
              M = node (12, 12), Q = node (6, 18))
"""

MATERIAL_DB = {
    "yu62": {"E": [172.4e9, 6.9e9, 6.9e9], "G": [3.45e9, 3.45e9, 1.38e9],
             "nu": [0.25, 0.25, 0.25], "rho": 1600.0},
}

W = 0.04
H = 0.01
THK = [H / 2, H / 2]
ANG = [90.0, 0.0]
MATS = ["yu62", "yu62"]

PMASS = 50.0
F0 = 10.0e3
T_RISE = 1.0e-3
T_END = 2.0e-3
DT = 1.0e-4
T_TOTAL = 0.05
T_REC = 0.0096

NX = 24
NZ_PLY = 4

XM = (W / 2, W / 2)
XQ = (H, W - H)
