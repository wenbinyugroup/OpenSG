"""make_plots_dyn3.py -- the dynamic comparison, THREE curves with the
same legend and marker conventions as the static four_way set:

    Reference 3-D FEA (Abaqus C3D20R)   solid red, circles
    Shear-refined (RM) Plate (OpenSG)   dashed black, filled squares
    Classical Plate (OpenSG)            dash-dot blue, LARGE OPEN boxes

Stress histories at the top-facesheet centre point (time + spectrum,
six components); the centre deflection w(t) keeps two curves (no
classical plate transient exists -- classical differs from RM only in
the RECOVERY, and w is a plate-solution quantity).

    python make_plots_dyn3.py [500us|2500us]

The first run moves the superseded two-curve PNGs of the tag folder
into <TAG>/pre_fix_backup/.

In:  ../opensg/<TAG>/topcenter_stress_t.dat            (corrected RM)
     ../opensg/<TAG>/topcenter_stress_t_classical.dat  (classical)
     ./<TAG>/topcenter_3dfea_t.dat     (Richardson onto midspan)
     ../abaqus/abaqus_2d_analysis/45pm_dynplate_<TAG>_ucen.csv (+50us
     fallbacks as in make_u3_time.py), ../abaqus/abaqus_3dfea/...
Out: <TAG>/time_S11..S23.png, freq_S11..S23.png, time_U3.png,
     freq_U3.png, dyn3_peaks.dat
"""
import os
import shutil
import sys
import time as _time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.normpath(os.path.join(HERE, "..", ".."))
TAG = "2500us"
for a in sys.argv[1:]:
    if a.strip().lower().endswith("us"):
        TAG = a.strip().lower()
OUT = os.path.join(HERE, TAG)
COMP = ["S11", "S22", "S33", "S12", "S13", "S23"]
NICE = {"S11": "11", "S22": "22", "S33": "33", "S12": "12",
        "S13": "13", "S23": "23"}

print("start :", _time.strftime("%H:%M:%S"))
print("tag   :", TAG)

# ---- park the superseded two-curve figures, once
bak = os.path.join(OUT, "pre_fix_backup")
if not os.path.isdir(bak):
    os.makedirs(bak)
    for f in os.listdir(OUT):
        if f.endswith(".png"):
            shutil.move(os.path.join(OUT, f), os.path.join(bak, f))
    print("moved the previous PNGs to pre_fix_backup/")

O = np.loadtxt(os.path.join(HERE, "..", "opensg", TAG,
                            "topcenter_stress_t.dat"))
C = np.loadtxt(os.path.join(HERE, "..", "opensg", TAG,
                            "topcenter_stress_t_classical.dat"))
D = np.loadtxt(os.path.join(OUT, "topcenter_3dfea_t.dat"))
print("stress: RM %d, classical %d, 3-D %d instants"
      % (len(O), len(C), len(D)))


def ucen(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    out = {}
    for nid in np.unique(d["node"]):
        m = d["node"] == nid
        t, iu = np.unique(np.round(d["t"][m], 12), return_index=True)
        out[int(nid)] = (t, d["U3"][m][iu])
    return out


PL = [os.path.join(HERE, "..", "abaqus", "abaqus_2d_analysis",
                   "45pm_dynplate_%s_ucen.csv" % TAG),
      os.path.join(HERE, "..", "abaqus", TAG, "abaqus_2d_analysis",
                   "45pm_dynplate_step_ucen.csv")]
D3 = [os.path.join(HERE, "..", "abaqus", "abaqus_3dfea",
                   "45pm_dyn3d_%s_ucen.csv" % TAG),
      os.path.join(HERE, "..", "abaqus", TAG, "abaqus_3dfea",
                   "45pm_dyn3d_step_ucen.csv"),
      os.path.join(B, "HC_pm45_dyn_clamped",
                   "45pm_dyn3d_step_ucen.csv")]
P = ucen(next(p for p in PL if os.path.exists(p)))
t_p, w_p = P[list(P)[0]]
S = ucen(next(p for p in D3 if os.path.exists(p)))
ntop = min(S, key=lambda n: S[n][1].mean())
t_w3, w_3 = S[ntop]

STY = [  # (label, style) -- the static four_way conventions
    ("Reference 3-D FEA (Abaqus C3D20R)",
     dict(ls="-", color="#c0392b", marker="o", ms=4.5, mfc="none",
          lw=1.6)),
    ("Shear-refined (RM) Plate (OpenSG)",
     dict(ls="--", color="k", marker="s", ms=3.0, lw=1.6)),
    ("Classical Plate (OpenSG)",
     dict(ls="-.", color="#2471a3", marker="s", ms=7.0, mfc="none",
          lw=1.4)),
]


def amp(t, y):
    """one-sided Hanning amplitude spectrum on the series' own grid"""
    dt = float(np.median(np.diff(t)))
    n = len(t)
    w = np.hanning(n)
    return (np.fft.rfftfreq(n, dt) * 1e-3,
            np.abs(np.fft.rfft((y - y.mean()) * w)) * 2.0
            / (n * w.mean()))


def draw(fn, series, ylab, kind):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for k, ((lab, st), (t, y)) in enumerate(zip(STY, series)):
        if y is None:
            continue
        me = max(len(t) // 26, 1)
        st = dict(st, markevery=(k * me // 3 + me // 6, me))
        if kind == "time":
            ax.plot(t * 1e6, y, label=lab, **st)
        else:
            fr, a = amp(t, y)
            ax.plot(fr, a, label=lab, **st)
    if kind == "time":
        ax.set_xlabel(r"$t$  ($\mu$s)")
    else:
        ax.set_xlim(0, 60)
        ax.set_xlabel("frequency  (kHz)")
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), ncol=3, columnspacing=1.0,
              handlelength=1.7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fn), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s/%s" % (TAG, fn))


rows = []
for j, cc in enumerate(COMP):
    ser = [(D[:, 0], D[:, 1 + j]), (O[:, 0], O[:, 1 + j]),
           (C[:, 0], C[:, 1 + j])]
    draw("time_%s.png" % cc, ser, r"$\sigma_{%s}$  (MPa)" % NICE[cc],
         "time")
    draw("freq_%s.png" % cc, ser, r"$|\sigma_{%s}|$  (MPa)" % NICE[cc],
         "freq")
    d3 = max(np.abs(D[:, 1 + j]).max(), 1e-30)
    rows.append((cc, np.abs(O[:, 1 + j]).max() / d3,
                 np.abs(C[:, 1 + j]).max() / d3))
ser_w = [(t_w3, w_3), (t_p, w_p), (t_p, None)]
draw("time_U3.png", ser_w, r"$w$  (mm)", "time")
draw("freq_U3.png", ser_w, r"$|w|$  (mm)", "freq")

with open(os.path.join(OUT, "dyn3_peaks.dat"), "w") as f:
    f.write("# peak |model| / |3-D| at the top-facesheet centre point,"
            " full %s window\n# %-4s %10s %12s\n"
            % (TAG, "comp", "RM", "classical"))
    for cc, r_rm, r_cl in rows:
        f.write("%-6s %10.3f %12.3f\n" % (cc, r_rm, r_cl))
        print("  %-4s peak ratio RM %.3f  classical %.3f"
              % (cc, r_rm, r_cl))
print("wrote %s/dyn3_peaks.dat" % TAG)
print("end   :", _time.strftime("%H:%M:%S"))
