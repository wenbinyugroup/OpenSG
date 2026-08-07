#!/bin/bash
# Routes A (msg_shell) and B (msg_solid new) for the four cases.
set -e
cd "$(dirname "$0")"
PY=$HOME/miniconda3/envs/opensg_2_0/bin/python
$PY make_inputs.py 2>&1 | grep -v -i "leaked\|nanobind"
for c in thick_iso thin_iso thick_m45 thin_m45; do
  echo "=== A $c ==="
  $PY run_shell.py $c 2>&1 | grep -v -i "leaked\|nanobind"
  echo "=== B $c ==="
  $PY run_solid_new.py $c 2>&1 | grep -v -i "leaked\|nanobind"
done
