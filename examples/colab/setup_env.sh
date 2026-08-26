#!/bin/bash
# setup_env.sh -- provision the ACTIVE python (conda env or system) for
# OpenSG GPU runs on Colab.  Run once per session, after activating the
# env you use:   bash setup_env.sh
# Installs the CUDA jax (the piece whose absence prints "CUDA-enabled
# jaxlib is not installed -- falling back to cpu") plus the solver and
# io dependencies, then verifies the GPU and float64.
set -e
pip install -q --upgrade "jax[cuda12]"
pip install -q pyamg pypardiso pyyaml scipy matplotlib fenics-basix flax
python - <<'PY'
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
dev = jax.devices()
print("devices :", dev)
print("dtype   :", jnp.zeros(3).dtype)
assert jnp.zeros(3).dtype == jnp.float64, "x64 OFF"
gpu = any(d.platform == "gpu" for d in dev)
print("GPU     :", "yes" if gpu else
      "NO -- is this a GPU runtime? (Runtime > Change runtime type)")
PY
echo "env ready"
