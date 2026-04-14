---
title: 'OpenSG: A FEniCSx-Based Implementation of the Mechanics of Structure Gene with Emphasis on Aperiodic Beams'

tags:
  - Python
  - wind energy
  - structural analysis
  - finite element method
  - composite materials
  - blade design
authors:
  - name: Kirk Bonney
    orcid: 0009-0006-2383-1634
    corresponding: true
    affiliation: 1
  - name: Akshat Bagla
    orcid: 0009-0003-3459-0731
    affiliation: 2
  - name: Ernesto Camarena
    orcid: 0000-0001-7835-6689
    affiliation: 1
  - name: Wenbin Yu
    orcid: 0000-0002-8065-7672
    affiliation: 2

affiliations:
 - name: Sandia National Laboratories, Albuquerque, NM, USA
   index: 1
 - name: Purdue University, West Lafayette, IN, USA
   index: 2

date: 21 August 2025
bibliography: paper.bib

---

# Summary

OpenSG is an open-source platform for multiscale structural mechanics built with a FEniCSx [@fenicsx] backend. It implements the Mechanics of Structure Genome (MSG) theory [@yu2019], a unified and rigorous framework for modeling advanced structures with general anisotropy and heterogeneity, including beams, plates, shells, and continuum structures.

OpenSG is a micromechanics code, which means it enables users to efficiently incorporate small-scale details into the global structural scale through constitutive modeling. This involves two primary steps, homogenization and dehomogenization. Homogenization is the process of obtaining homogenized structural properties whereas, dehomogenization is recovering the local stress and displacement fields. These operations are performed over a user-defined domain called the Structure Gene (SG). An SG can be 1-dimensional (line elements), 2-dimensional (quadrilaterals or triangular elements), or 3-dimensional (hexahedron or tetrahedron elements). Using SGs, OpenSG produces structural properties in terms of Cauchy continuum model, plate/shell model, or beam model. Conventional finite element analysis tools can utilize these properties to output global structural responses. OpenSG can use these responses to compute small-scale stresses and deformations throughout the SG. In this way, OpenSG provides an accurate, fast, and versatile platform for analyzing structural profiles with key small-scale details like aircraft, wind turbine blades, and additively manufactured parts.

# Statement of need

No existing open-source tool provides MSG theory for Timoshenko beam modeling of aperiodic solid SGs, and shell cross-sections with local buckling predictions. OpenSG fills this gap with an open-source MSG implementation [@opensg] that supports aperiodic beam SGs composed of shell [@opensg_prismatic_shell_to_beam; @opensg_RM_shell] or solid [@opensg_prismatic_shell_to_beam] elements. This is especially relevant for nonprismatic structures such as wind turbine blades, tapered rods, propellers, and aircraft wings where 3D phenomena like local panel buckling cannot be captured by beam-only approaches.

# State of the Field

Several tools exist for computing structural properties of composite beams and blades. SwiftComp [@swiftcomp] is the primary reference implementation of MSG theory, unifying VAMUCH [@vamuch] for material unit cells, VAPAS [@vapas] for plates and shells, and VABS [@Yu2002Timoshenko] for beams. However, SwiftComp is commercial and closed-source, limiting reproducibility and community extensibility. BECAS [@becas], ANBA4 [@anba4], and PreComp [@precomp] are open-source beam cross-section analysis tools widely used in wind energy, but they are restricted to beam models and do not implement the MSG framework.

# Software Design

OpenSG is built on FEniCSx [@fenicsx], a modern finite element framework whose Unified Form Language (UFL) allows the MSG governing equations to be expressed in a form that closely mirrors their mathematical derivations. This keeps the implementation auditable and accessible to researchers working directly from the MSG literature. FEniCSx's comprehensive Python API and active development community make it a natural foundation for an extensible, open-source structural analysis tool.

The codebase is organized around element formulation type: shell and solid element formulations are implemented in `opensg.core.shell` and `opensg.core.solid` respectively, with shared utilities for material property assignment and stiffness matrix assembly in `opensg.utils`. Material properties and composite layup definitions are supplied via YAML input files and parsed through `opensg.io`. Mesh generation is managed through `opensg.mesh`, which interfaces with external tools such as Gmsh to prepare the necessary data structures to apply the MSG algorithms.

# Research Impact Statement

The development team has applied OpenSG to advance structural analysis of composite structures, including 3D shell-to-beam modeling [@opensg_prismatic_shell_to_beam] and wind turbine blade modeling [@opensg_timoshenko]. These applications highlight the potential of OpenSG to enable accurate, scalable, and open-source solutions for structural modeling challenges across wind energy and beyond.

The repository includes detailed documentation and multiple Python notebook examples to support external adoption. The test suite includes regression tests for shell and solid stiffness computations against stored baseline results, ensuring reproducibility and providing a clear record of how outputs evolve as the source code is updated. OpenSG is part of the [WETO Software Stack](https://github.com/NatLabRockies/WETOStack), a collection of integrated modeling tools for wind energy research; integration with other WETO codes allows researchers to incorporate OpenSG directly into existing wind energy workflows without requiring standalone preprocessing pipelines.

# AI usage disclosure

No generative AI tools were used in the development of this software or the preparation
of this manuscript.

# Acknowledgements
OpenSG has been developed through a collaboration between Sandia National Laboratories and Purdue University within the Holistic, Multi-Fidelity Wind Farm Design Optimization and Model Coordination project under the U.S. Department of Energy Wind Energy Technologies Office. It is currently being evaluated for its suitability to transform wind turbine blade design, mainly due to its ability to incorporate high-fidelity structural details into efficient beam models.

Sandia National Laboratories is a multimission laboratory managed and operated by National Technology and Engineering Solutions of Sandia, LLC., a wholly owned subsidiary of Honeywell International, Inc., for the U.S. Department of Energy's National Nuclear Security Administration under contract DE-NA-0003525.

# References 
