#!/bin/bash
# ROUTE C -- the OLD JAX_BICGoptimize single-script driver, n_model = 3.
# Its fe_jax lives in ~/OpenSG_2.0 (model-aware periodic_multiscale); jax_env is
# the interpreter whose jax is new enough for the driver's jit decorators.
# Only the two ISOTROPIC cases: a .sc cannot carry per-element material frames.
set -e
cd "$(dirname "$0")"
export PYTHONPATH=$HOME/OpenSG_2.0:$HOME/OpenSG_2.0/msg_akshat
export CELL_AREA=1.0
for c in thick_iso thin_iso; do
  echo "=== C $c ==="
  CASE=$c $HOME/miniconda3/envs/jax_env/bin/python solid_props_bicg.py 2>&1 \
      | grep -v -i "leaked\|nanobind"
done
