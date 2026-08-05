"""ff_to_yaml.py -- BeamDyn/VABS beam-FF .dat -> per-station yaml, the beam
analog of rm_plate_1D.helper.sc_to_yaml (mesh) and abq2ff (plate FF field).

The .dat this reads is the spanwise beam force/moment table a 1-D solver
hands the cross-section recovery:

    # eta F1 F2 F3 M1 M2 M3          (VABS order, header lines start with #)
    0.00  ...                        eta = r/R, one row per station

Outputs <base>.yaml with the stations as a list plus the per-station block
the dehom driver consumes:

    convention: {order: [F1,F2,F3,M1,M2,M3], frame: "RM center-ref",
                 units: {force: N, moment: N.m}}
    stations:
      - {row: 0, eta: 0.0, FF: [...]}
      ...

Use:  from opensg_shell.helper.ff_to_yaml import convert, station_ff
      d = convert("ff51_rmc_reform.dat")       # writes ff51_rmc_reform.yaml
      FF, eta = station_ff("ff51_rmc_reform.yaml", eta=0.20)
"""
import os

import numpy as np
import yaml


def read_ff(path):
    """(eta (n,), FF (n,6)) from the .dat table; '#' lines are the header."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.shape[1] < 7:
        raise ValueError("%s: expected 7 columns (eta F1 F2 F3 M1 M2 M3), got %d"
                         % (path, raw.shape[1]))
    return raw[:, 0], raw[:, 1:7]


def convert(dat_path, out_base=None, frame="RM center-ref"):
    """Read the FF .dat, write <base>.yaml, return the dict."""
    if out_base is None:
        out_base = os.path.splitext(dat_path)[0]
    eta, FF = read_ff(dat_path)
    d = {"source": os.path.basename(dat_path),
         "convention": {"order": ["F1", "F2", "F3", "M1", "M2", "M3"],
                        "frame": frame,
                        "units": {"force": "N", "moment": "N.m"}},
         "stations": [{"row": int(i), "eta": float(eta[i]),
                       "FF": [float(v) for v in FF[i]]}
                      for i in range(len(eta))]}
    with open(out_base + ".yaml", "w") as f:
        yaml.safe_dump(d, f, sort_keys=False, default_flow_style=None)
    print("ff_to_yaml: %d stations (eta %.3f..%.3f) -> %s.yaml"
          % (len(eta), eta.min(), eta.max(), out_base))
    return d


def station_ff(yaml_path, eta=None, row=None):
    """The (6,) FF of one station, by eta (nearest) or by row index."""
    d = yaml.safe_load(open(yaml_path))
    st = d["stations"]
    if row is None:
        if eta is None:
            raise ValueError("give eta or row")
        row = int(np.argmin([abs(s["eta"] - eta) for s in st]))
    return np.asarray(st[row]["FF"], float), float(st[row]["eta"])
