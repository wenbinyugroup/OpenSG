#!/bin/bash
# Everything: decks, route A (msg_shell), route B (msg_solid new), route C
# (msg_solid old, isotropic cases only), the +45 sign-convention probe, the
# tables.  Route C needs a different interpreter and PYTHONPATH -- see run_c.sh.
set -e
cd "$(dirname "$0")"
PY=$HOME/miniconda3/envs/opensg_2_0/bin/python
F='grep -v -i leaked'

$PY make_inputs.py 2>&1 | grep -v -i "leaked\|nanobind"
for c in thick_iso thin_iso thick_m45 thin_m45; do
  echo "=== A $c ==="; $PY run_shell.py $c      2>&1 | grep -v -i "leaked\|nanobind"
  echo "=== B $c ==="; $PY run_solid_new.py $c  2>&1 | grep -v -i "leaked\|nanobind"
done
# fibre-angle SIGN-CONVENTION probe: route B re-run at +45 (see README.md)
for c in thick_m45 thin_m45; do
  echo "=== B $c (+45 probe) ==="
  $PY run_solid_new.py $c 45 2>&1 | grep -v -i "leaked\|nanobind"
done
bash run_c.sh
$PY make_tables.py 2>&1 | grep -v -i "leaked\|nanobind"
