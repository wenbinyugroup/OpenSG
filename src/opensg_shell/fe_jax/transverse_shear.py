"""
Re-export of the laminate transverse-shear stiffness (the RM 8x8 G block).

The canonical implementation now lives in the package at
``opensg_jax/fe_jax/msg_transverse_shear.py`` so that the plate subroutine
``compute_ABD_matrix(..., shear_refined=True)`` can assemble the 8x8 RM plate
stiffness directly.  This thin module exposes it under the package import path
(``from opensg_jax.fe_jax.transverse_shear import transverse_shear_stiffness``); the
``coupled=True`` (MSG, coupling-aware) form is the default.
"""
import os
import sys
from .msg_transverse_shear import (   # noqa: E402,F401
    _ply_Q_and_G,
    transverse_shear_stiffness,
    plate_8x8,
)


if __name__ == "__main__":
    import numpy as np
    E, nu, h = 3.44e9, 0.3, 0.2
    G = E/(2*(1+nu))
    mat = {"iso": {"E": [E, E, E], "G": [G, G, G], "nu": [nu, nu, nu]}}
    Gmat, rec, _ = transverse_shear_stiffness([h], [0.0], ["iso"], mat)
    print("Isotropic plate transverse-shear stiffness:")
    print(f"  F (MSG)   = {Gmat[0,0]:.6e}")
    print(f"  (5/6) G h = {5/6*G*h:.6e}")
    print(f"  ratio     = {Gmat[0,0]/(5/6*G*h):.6f}   (should be 1.000)")
