#!/usr/bin/env bash
# .sg -> OpenSG 2-D solid YAML (with elementOrientations), then checks + plots.
set -u
D=$HOME/OpenSG-2.0/tests/08072026_square_shell_mesh
PY=$HOME/miniconda3/envs/opensg_2_0/bin/python
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

echo "=== convert_sg_to_yaml ==="
$PY $HOME/OpenSG_io/scripts/convert_sg_to_yaml.py \
    $D/prevabs/square_tube.sg  $D/square_tube_2dsolid.yaml
echo
echo "=== checks + plots ==="
$PY $D/check_and_plot.py
