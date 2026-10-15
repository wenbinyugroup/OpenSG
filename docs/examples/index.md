# Examples

Each example is a complete OpenSG run on one structure gene (SG): the input yaml, a short driver
script and the output files the run produces, all in one folder under `examples/`. Every example
page follows the same ladder: problem description, input, run, result and files. The numbers on a
page are copied from the `.out` and comparison files shipped in its folder, so you can check a run
of your own against the page entry by entry.

The examples are grouped by engine. `opensg_shell` takes SGs of shell elements: a 1-D shell SG is
the contour of a cross-section or of a periodic cell, and a 3-D shell SG is a surface mesh.
`opensg_solid` takes SGs of solid elements: a 1-D SG is a through-thickness stack, a 2-D SG is a
cross-section or a cell, and a 3-D SG is a unit cell. The key-by-key description of the input is in
{doc}`../user_guide/index`.

## Overview

{numref}`tab-examples-overview` lists the thirteen example pages with the SG each one uses, the
engine that solves it, the macro model it produces and the reference its result is compared with.

% source: yaml headers (msg:, n_model:, refined:) of the example folders under examples/OpenSG_shell/ and examples/OpenSG-solid/; docs/tutorials/msg_shell_beam.md; docs/tutorials/solid_props_shell_sg.md; docs/tutorials/solid_props_3d_sg.md; docs/tutorials/taper_segment_aperiodic.md; docs/tutorials/windio_blade_pipeline.md; docs/tutorials/msg_solid_plate.md; docs/tutorials/msg_solid_beam_and_solid.md; examples/OpenSG_shell/windio/README.md; examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section/square_section_isotropic/ (folder listing) and docs/examples/shell_solid_cellular.md ("Isotropic square tube"); docs/examples/shell_beam_recovery.md (step 2)
```{list-table} The examples
:header-rows: 1
:name: tab-examples-overview

* - Example
  - Structure gene
  - Engine
  - Macro model
  - Compared with
* - {doc}`shell_beam_cross_section`
  - 1-D shell SG: the IEA-22 blade cross-section at $r/R = 0.20$
  - `opensg_shell`
  - Timoshenko beam $6\times6$
  - self-check: extension does not couple to transverse shear or twist
* - {doc}`shell_beam_recovery`
  - the same 1-D shell SG
  - `opensg_shell` with `opensg_solid.rm_plate_1D`
  - Timoshenko beam; recovered 3-D wall stress
  - VABS 2-D solid model of the same section
* - {doc}`shell_solid_cellular`
  - 1-D shell SG of a periodic cell: cellular solid at $+15^\circ$ and $-15^\circ$, hierarchical square, composite square, and an isotropic square tube
  - `opensg_shell`
  - 3-D solid $6\times6$
  - Deo and Yu (2023): the paper's thin-walled model and its SwiftComp solid model
* - {doc}`shell_solid_3d_tpms`
  - 3-D shell SG: the Schwarz-P surface of a unit cell
  - `opensg_shell`
  - 3-D solid $6\times6$
  - aperiodic boundaries on the same yaml; the solid route on the same surface
* - {doc}`shell_taper_segment`
  - 3-D shell SG: a beam segment meshed with four-node quads (prismatic tube, BAR-URC segment 12)
  - `opensg_shell`
  - Timoshenko beam $6\times6$
  - prismatic tube: the boundary ring's own $6\times6$
* - {doc}`blade_iea_beam`
  - 1-D shell SG per span station of the IEA-22 blade (51 stations)
  - `opensg_shell` with `opensg_solid.rm_plate_1D`
  - Timoshenko beam $6\times6$; recovered 3-D wall stress
  - VABS 2-D solid model of station `iea_s10`; thickness integral of the recovered transverse shear
* - {doc}`blade_windio_pipeline`
  - 1-D shell SG per span station, generated from the windIO file `IEA-22-280-RWT.yaml`
  - `opensg_shell`
  - Timoshenko beam $6\times6$ and $6\times6$ mass; BeamDyn blade; recovered 3-D wall stress
  - VABS `.K` files of the same blade meshed as 2-D solid sections
* - {doc}`solid_plate_1d_layup`
  - 1-D solid SG: a sandwich laminate through the thickness
  - `opensg_solid`
  - plate ABD $6\times6$ (extension $A$, extension-bending coupling $B$ and bending $D$ stiffness) or Reissner-Mindlin ABDG $8\times8$ (the ABD plus the $2\times2$ transverse-shear block $G$)
  - classical lamination theory; the isotropic shear anchor $\tfrac56 G h$
* - {doc}`solid_plate_1d_recovery`
  - the same 1-D solid SG
  - `opensg_solid`
  - Reissner-Mindlin plate; recovered 3-D ply stress
  - Pagano cylindrical-bending solutions; the cases of Yu, Hodges and Volovoi (2003)
* - {doc}`solid_plate_2d_honeycomb`
  - 2-D solid SG: the honeycomb sandwich cell `RHC_SW_2UC_45`
  - `opensg_solid`
  - plate ABD $6\times6$ or Reissner-Mindlin ABDG $8\times8$; recovered 3-D fields
  - the cell's measured period and its mid-plane symmetry
* - {doc}`solid_equivalent_2d`
  - 2-D solid SG: a square unidirectional fibre/matrix cell
  - `opensg_solid`
  - 3-D solid $6\times6$
  - Table II Model-3 of the ASC-23 OpenSG-Solid paper
* - {doc}`solid_equivalent_3d_tpms`
  - 3-D solid SG: Schwarz-P sheets meshed with linear tetrahedra
  - `opensg_solid`
  - 3-D solid $6\times6$
  - SwiftComp `.K` results of the same meshes
* - {doc}`solid_beam_2d`
  - 2-D solid SG: the honeycomb sandwich cross-section `RHC_SW_2UC_45`
  - `opensg_solid`
  - Timoshenko beam $6\times6$ with the Euler-Bernoulli $4\times4$; recovered 3-D fields
  - the Euler-Bernoulli $4\times4$ of the same solve
```

## How to run an example

Every example runs in place. Change into its folder and run the command or the driver script
listed in {numref}`tab-examples-run`; the example owns every path, so run it from inside the
folder. When the yaml header carries `msg:`, the one command `opensg <sg.yaml>` is enough:

% source: docs/tutorials/msg_shell_beam.md (Example 1, "Run it")
```bash
cd examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section
opensg iea_s10_shell.yaml --center
```

`opensg <sg.yaml>` homogenizes, `opensg <sg.yaml> D` homogenizes and then recovers the local
fields, and `--center` (shell SG only) takes the contour as the laminate mid-surface instead of the
outer mold line. The driver script of a folder runs the analysis through the Python API; in the
first example it does the same as the command and adds the two figures the command does not emit:

% source: docs/tutorials/msg_shell_beam.md (Example 1, "Run it", lines 150-154)
```bash
python beam_homo_shell.py
```

The command line options are described in {doc}`../getting_started/running`.

% source: the bash blocks of docs/tutorials/msg_shell_beam.md, solid_props_3d_sg.md, taper_segment_aperiodic.md, windio_blade_pipeline.md, msg_solid_plate.md, msg_solid_beam_and_solid.md; examples/OpenSG_shell/windio/README.md; the driver scripts present in each example folder (square_section_isotropic/solid_props_msg_shell.py from the folder listing and docs/examples/shell_solid_cellular.md "Files")
```{list-table} Folders and commands of the examples
:header-rows: 1
:name: tab-examples-run

* - Example
  - Folder
  - Command
* - {doc}`shell_beam_cross_section`
  - `examples/OpenSG_shell/1_get_beam_props_from_shell_cross_section`
  - `opensg iea_s10_shell.yaml --center` or `python beam_homo_shell.py`
* - {doc}`shell_beam_recovery`
  - `examples/OpenSG_shell/2_get_beam_dehom_from_shell_cross_section`
  - `opensg iea_s10_shell.yaml D --center` or `python beam_dehom_shell.py`
* - {doc}`shell_solid_cellular`
  - `examples/OpenSG_shell/3_get_solid_props_from_shell_cross_section`
  - `python deo_cellular_solid_props.py` in `deo_cellular_solid/`, `python deo_square_solid_props.py` in `deo_hierarchical_square/`, `python deo_composite_square_props.py` in `deo_composite_square/`, `python solid_props_msg_shell.py` in `square_section_isotropic/`
* - {doc}`shell_solid_3d_tpms`
  - `examples/OpenSG_shell/4_get_solid_props_from_shell_3D_SG`
  - `opensg schwarz_p_3Dshell.yaml` or `python schwarz_solid_props.py`
* - {doc}`shell_taper_segment`
  - `examples/OpenSG_shell/5_taper_segment_aperiodic_boundaries`
  - `python run_taper_segment.py` (prismatic tube), `python run_bar_urc_segment.py` (BAR-URC segment)
* - {doc}`blade_iea_beam`
  - `examples/OpenSG_shell/IEA_blade_beam`
  - `python 1_homo_beam_props.py`, `python 2_dehom_v2_vabs.py`, `python 3_q_consistency_check.py`, `python 4_sweep_51_stations.py`
* - {doc}`blade_windio_pipeline`
  - `examples/OpenSG_shell/windio`
  - `python 1_windio_to_cross_sections.py` to `python 7_verify_blade_timo_vs_K.py`, in the order of their numbers
* - {doc}`solid_plate_1d_layup`
  - `examples/OpenSG-solid/1_get_plate_props_from_1DSG`
  - `python 1d_sg.py`
* - {doc}`solid_plate_1d_recovery`
  - `examples/OpenSG-solid/2_get_plate_stress_from_1DSG`
  - `python rm_dehom.py` or `python rm_dehom_ff.py ex5_shell_S8R_field.ff`
* - {doc}`solid_plate_2d_honeycomb`
  - `examples/OpenSG-solid/3_get_plate_props_from_2D_SG` and `examples/OpenSG-solid/4_get_plate_dehom_from_2DSG`
  - `opensg RHC_SW_2UC_45.yaml` in the first, `python plate_dehom_2dsg.py` in the second
* - {doc}`solid_equivalent_2d`
  - `examples/OpenSG-solid/5_get_solid_props_from_SG`
  - `python make_udcomp_yaml.py`, then `python solid_homo_2dsg.py`
* - {doc}`solid_equivalent_3d_tpms`
  - `examples/OpenSG-solid/6_get_solid_props_from_3D_SG`
  - `python run_sample_1.py` in `sample_1/`, `python run_sample_2.py` in `sample_2/`
* - {doc}`solid_beam_2d`
  - `examples/OpenSG-solid/7_get_beam_props_from_SG` and `examples/OpenSG-solid/8_get_beam_dehom_from_SG`
  - `python beam_homo_sg.py` in the first, `python beam_dehom_sg.py` in the second
```

:::{note}
Most folders ship the `.out` files of their homogenization, so the stiffness tables of a page can
be compared without a run. A run writes its output files inside the example folder.
:::

% source: docs/tutorials/windio_blade_pipeline.md (section 3); examples/OpenSG_shell/windio/README.md (Run)
:::{tip}
Two blade scripts need inputs from outside OpenSG. `4_sweep_51_stations.py` in `IEA_blade_beam`
reads the 51 station yamls from `yaml_dir`, the output of OpenSG_io; point it at your own
directory. `5_run_beamdyn_extract_ff.py` in `windio` needs `beamdyn_driver` (conda-forge
openfast).
:::

% source: README.md (Examples)
The folder `examples/pynumad/` holds the `opensg.read(blade.yaml)` Blade object for
station-by-station blade work.

## Example pages

```{toctree}
:maxdepth: 1
:caption: Shell structure genes

shell_beam_cross_section
shell_beam_recovery
shell_solid_cellular
shell_solid_3d_tpms
shell_taper_segment
blade_iea_beam
blade_windio_pipeline
```

```{toctree}
:maxdepth: 1
:caption: Solid structure genes

solid_plate_1d_layup
solid_plate_1d_recovery
solid_plate_2d_honeycomb
solid_equivalent_2d
solid_equivalent_3d_tpms
solid_beam_2d
```
