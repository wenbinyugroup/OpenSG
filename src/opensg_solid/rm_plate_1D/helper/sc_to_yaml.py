"""MOVED -- the SwiftComp .sc -> yaml converter now lives in
opensg_solid.helper.sc_to_yaml (the solid-side `helper` subpackage, the
opensg_shell.helper analog); this module re-exports it unchanged so the
historical import path keeps working."""
from opensg_solid.helper.sc_to_yaml import *          # noqa: F401,F403
from opensg_solid.helper.sc_to_yaml import (          # noqa: F401
    bbox_measure, convert, default_n_model, dump_materials, header_omega,
    read_sc, write_msh, write_yaml_file, yaml_materials)
