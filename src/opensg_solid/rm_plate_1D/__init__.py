"""OpenSG-RM plate stack: MSG 1-D SG homogenization (8x8 ABDG) +
3-D dehomogenization/recovery.  See msg_rm_plate_README.md.

The x64 switch below is LOAD-BEARING: the warping ladder, the KKT solves
and the V2L face enforcement are validated in double precision -- without
it every gate drifts at float32 eps (faces ~1e-3 instead of ~1e-11)."""
import jax

jax.config.update("jax_enable_x64", True)
