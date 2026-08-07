#!/usr/bin/env bash
# Build the square thin-walled tube 2-D cross section with PreVABS and convert
# the VABS .sg mesh to the OpenSG 2-D solid YAML (with elementOrientations).
set -u
D=$HOME/OpenSG-2.0/tests/08072026_square_shell_mesh
PVDIR=$HOME/OpenSG_io/third_party/prevabs_bin/prevabs-v2.1.0-preview.20260508.3-linux-rhel9-x64
PV=$PVDIR/prevabs
PY=$HOME/miniconda3/envs/opensg_2_0/bin/python
export LD_LIBRARY_PATH=$PVDIR:$HOME/miniconda3/envs/opensg_2_0/lib:${LD_LIBRARY_PATH:-}

cd $D/prevabs || exit 1
echo "=== prevabs -i square_tube.xml --vabs --hm ==="
timeout 600 $PV -i square_tube.xml --vabs --hm
echo "exit=$?"
echo
echo "=== outputs ==="
ls -la $D/prevabs
