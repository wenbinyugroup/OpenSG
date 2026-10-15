# References

This page lists the publication to cite when you use OpenSG, the theory papers that the
documentation refers to, and the version history of the code.

## Citing OpenSG

If you use OpenSG, cite the conference paper that introduces it:

% source: CITATION.cff (preferred-citation)
A. Bagla, W. Yu and E. Camarena, "OpenSG: An Open-Source Multiscale Constitutive Modeling Code
for Composite Structures and Materials", *Proceedings of the American Society for Composites,
38th Technical Conference*, 2023. doi:
[10.12783/asc38/36715](https://doi.org/10.12783/asc38/36715).

% source: CITATION.cff (preferred-citation)
```bibtex
@inproceedings{bagla2023opensg,
  title     = {OpenSG: An Open-Source Multiscale Constitutive Modeling Code for Composite Structures and Materials},
  author    = {Bagla, Akshat and Yu, Wenbin and Camarena, Ernesto},
  booktitle = {Proceedings of the American Society for Composites, 38th Technical Conference},
  year      = {2023},
  doi       = {10.12783/asc38/36715}
}
```

The software itself is described by the entry of {numref}`tab-ref-software`. The "Cite this
repository" button of the GitHub repository formats the citation from the same file,
`CITATION.cff`.

% source: CITATION.cff
```{list-table} The software entry of CITATION.cff
:header-rows: 1
:name: tab-ref-software

* - Field
  - Value
* - Title
  - OpenSG: multiscale structural mechanics by the Mechanics of Structure Genome
* - Authors
  - Akshat Bagla (Purdue University), Wenbin Yu (Purdue University), Ernesto Camarena (Sandia National Laboratories)
* - Version
  - 2.1.0
* - Released
  - 2026-10-09
* - Repository
  - <https://github.com/wenbinyugroup/OpenSG>
* - License
  - Apache-2.0
```

The finite-element core `fe_jax`, included at `src/fe_jax`, is by Keith Ballard (Air Force
Research Laboratory): the core of [fea-in-jax](https://github.com/KeithBallard/fea-in-jax), which
provides the basis and quadrature tables, the periodic assembly map and the matrix-free solvers
used by both engines. Distribution Statement A. Approved for public release: distribution is
unlimited. Case #: AFRL-2025-4644.

## Theory papers

The Mechanics of Structure Genome (MSG) plate and thin-walled formulations that OpenSG implements
are cited in the theory pages and the examples as listed in {numref}`tab-ref-papers`. Each entry
is given as it appears there.

% source: docs/tutorials/msg_solid_plate.md (section A); docs/tutorials/msg_shell_beam.md ("The element and the wall law"); docs/theory/constitutive.md (section 6); docs/theory/rm_plate_1d_notes.md (header); docs/tutorials/solid_props_shell_sg.md ("Validation")
```{list-table} Theory papers cited in this documentation
:header-rows: 1
:name: tab-ref-papers

* - Reference
  - Used for
  - Cited in
* - Yu, Hodges and Volovoi, *Computers & Structures* 81 (2003) 439–454.
  - the through-thickness Reissner-Mindlin (RM) plate structure gene of `rm_plate_msg`: the transverse-shear block $G$ (Eq. 61), the first-order recovery (Eq. 63) and the second-order recovery (Eq. 66)
  - {doc}`theory/constitutive`, {doc}`theory/rm_plate_1d_notes`, {doc}`examples/solid_plate_1d_layup`, {doc}`examples/solid_plate_1d_recovery`, {doc}`examples/shell_beam_cross_section`
* - IJSS 39:5185 (Eqs. 49–55) and CMAME 191:5087 (Eqs. 70–77)
  - the same construction, in the equation numbers of those papers
  - {doc}`theory/rm_plate_1d_notes`
* - A. Deo and W. Yu, *Equivalent 3D properties of thin-walled composite structures using mechanics of structure genome*, Mechanics of Advanced Materials and Structures 30(9), 2023, 1737–1748.
  - the four cellular and thin-walled cross-sections, with the paper's thin-walled (MSG-TW) and MSG solid columns as the references
  - {doc}`examples/shell_solid_cellular`
```

## Version history

{numref}`tab-ref-versions` lists the released versions. The current line, OpenSG 2.x, runs on JAX
and replaces the 1.0 code, which was written on FEniCSx.

% source: README.md (Versions, Documentation); CITATION.cff (version, date-released); pyproject.toml (version); annotations of the release tags (`git tag -n v2.0.0 v2.1.0` in the repository)
```{list-table} Versions of OpenSG
:header-rows: 1
:name: tab-ref-versions

* - Version
  - Branch and tag
  - Content
* - 2.1.0
  - `main`, tag `v2.1.0`
  - `opensg` 2.1.0, released 2026-10-09: `elementOrientations` applied on every homogenization route (command line included); `pypardiso` declared. This documentation describes this version.
* - 2.0.0
  - `main`, tag `v2.0.0`
  - OpenSG 2.0.0 (the JAX line), which replaces the 1.0 code.
* - 1.0.0
  - `opensg/fenicsx/v1`, tag `v1.0.0`
  - `opensg` 1.0.0, OpenSG 1.0 on FEniCSx (frozen).
```

The branch `opensg/fenicsx/v1.2` holds the later FEniCSx code. The documentation of OpenSG 2.x is
hosted at [wenbinyugroup.github.io/OpenSG/OpenSG_v2_0](https://wenbinyugroup.github.io/OpenSG/OpenSG_v2_0/),
beside the documentation of OpenSG 1.0 at
[wenbinyugroup.github.io/OpenSG](https://wenbinyugroup.github.io/OpenSG/).
