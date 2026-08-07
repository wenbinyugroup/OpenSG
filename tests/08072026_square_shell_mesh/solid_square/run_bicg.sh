#!/bin/bash
# Old JAX_BICGoptimize single-script driver, n_model = 3, on square_tube.sc.
# Its fe_jax lives in ~/OpenSG_2.0 (model-aware periodic_multiscale); jax_env is
# the interpreter whose jax is new enough for the driver's jit decorators.
set -e
cd "$(dirname "$0")"
export PYTHONPATH=$HOME/OpenSG_2.0:$HOME/OpenSG_2.0/msg_akshat
$HOME/miniconda3/envs/jax_env/bin/python solid_props_bicg.py
