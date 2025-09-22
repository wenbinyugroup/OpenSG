---
title: 'OpenSG: A FEniCS-Based Implementation of Mechanics of Structure Gene with Empahsis on Aperiodic Beams'

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
    orcid: 0000-0000-0000-0000
    affiliation: 2
  - name: Ernesto Camarena
    orcid: 0000-0001-7835-6689
    affiliation: 1

affiliations:
 - name: Sandia National Laboratories, Albuquerque, NM, USA
   index: 1
 - name: Purdue University, West Lafayette, IN, USA
   index: 2

date: 21 August 2025
bibliography: paper.bib

---

# Summary

OpenSG is an open-source platform for multiscale structural mechanics using FEniCSx backend which implements the Mechanics of Structure Genome (MSG) theory [@yu2019]. This unified and revolutionary theory provides a rigorous and systematic approach to modeling advanced structures featuring general anisotropy and heterogeneity, including beams, plates, shells, and continuum structures.
 Gene (S
Being a micromechanics code, OpenSG is useful for incorporating micro- and mesoscale features into other analysis programs via homogenization and dehomogenization. Homogenization is the process of obtaining homogenized structural properties whereas, dehomogenization is recovering the local stress and displacement fields. The homogenization and dehomogenization take place over the user-defined domain called the Structure Gene (SG), which is the mathematical building block of structure being studied. The SG serves as the mathematical building block to define the heterogeneity and anisotropy. It can be meshed with either 1D line elements, 2D quadrilateral or triangular elements, or 3D hexahedron or tetrahedron elements. The SG can then undergo homogenization to obtain structural properties in terms of Cauchy continuum model, plate/shell model, or beam model, providing accurate, fast, and precise analysis for complex structural profiles like wind turbine blades. Thus, enabling the research community to leverage the advantages of MSG theory for generating complex models for specific applications.

OpenSG has been developed within the Holistic, Multi-Fidelity Wind Farm Design Optimization and Model Coordination project under the U.S. DOE Wind Energy Technologies Office. It is currently being evaluated for its suitability to transform wind turbine blade design, mainly due to its ability to incorporate high-fidelity structural details into efficient beam models. Recent research has already demonstrated the utility of OpenSG in advancing structural analysis of composite structures, including 3D shell-to-beam modeling [@opensg_tapered_solid] and wind turbine blade modeling [@opensg_timoshenko]. These applications highlight the potential of OpenSG to enable accurate, scalable, and open-source solutions for structural modeling challenges across wind energy and beyond.

# Statement of need

The MSG theory was first implemented in SwiftComp [@swiftcomp], a commercial code. It offers unification of the following codes: VAMUCH for unit cells of materials [@vamuch], VAPAS [@vapas] for plates and shells, and VABS [@cesnik1997vabs] for beams. In addition to replicating the core of SwiftComp's capabilities [@opensg], OpenSG extends the MSG theory to allow for aperiodic beam SGs made of either shell [@opensg_prismatic_shell_to_beam; @opensg_RM_shell] or sold [@opensg_tapered_solid] elements. This is particularly useful for modeling nonprismatic structures and for accounting for 3D phenomenon, such as panel buckling. Some examples of nonprismatic structures are airplane wings, propellers, offshore jacketed structures, wind blades, tapered tubes and rods, non prismatic bridges. While there is similar software such as BECAS and PreComp which also compute beam properties, OpenSG is unique in that it additionally computes shell and solid models and utilizes MSG as the underlying theory.

# Acknowledgements
Sandia National Laboratories is a multimission laboratory managed and operated by National Technology and Engineering Solutions of Sandia, LLC., a wholly owned subsidiary of Honeywell International, Inc., for the U.S. Department of Energy’s National Nuclear Security Administration under contract DE-NA-0003525.

# References 
