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

OpenSG is a micromechanics code, which means it enables users to effeciently incorporate small-scale details into the global structural scale. This involves two primary steps, homogenization and dehomogenization. Homogenization is the process of obtaining homogenized structural properties whereas, dehomogenization is recovering the local stress and displacement fields. These operations are performed over a user-defined domain called the Structure Gene (SG). An SG can be 1-dimensional (line elements), 2-dimensional (quadrilaterals or triangular elements), or 3-dimensional (hexahedron or tetrahedron elements). Using SGs, OpenSG produces structural properties in terms of Cauchy continuum model, plate/shell model, or beam model. Conventional finite element analysis tools can utilize these properties to output global structural responses. OpenSG can use these responses to compute small-scale stresses and deformations throughout the SG. In this way, OpenSG provides an accurate, fast, and versatile platform for analyzing structural profiles with key small-scale details like aircraft, wind turbine blades, and additively manufactured parts.

# Statement of need

The MSG theory was first implemented in SwiftComp [@swiftcomp], a commercial code. It offers unification of the following codes: VAMUCH for unit cells of materials [@vamuch], VAPAS [@vapas] for plates and shells, and VABS [@Yu2002Timoshenko] for beams. In addition to replicating the core of SwiftComp's capabilities [@opensg], OpenSG extends the MSG theory to allow for aperiodic beam SGs made of either shell [@opensg_prismatic_shell_to_beam; @opensg_RM_shell] or solid [@opensg_prismatic_shell_to_beam] elements. This is particularly useful for modeling nonprismatic structures and for accounting for 3D phenomenon, such as panel buckling. Some examples of nonprismatic structures are airplane wings, propellers, offshore jacketed structures, wind blades, tapered tubes and rods, non prismatic bridges. 

Compared to other beam property computation tools such as BECAS [@becas] and PreComp [@precomp], OpenSG is unique in that it goes beyond beam-only models, also supporting shell and solid formulations within a single MSG-based framework.

Recent research has already demonstrated the utility of OpenSG in advancing structural analysis of composite structures, including 3D shell-to-beam modeling [@opensg_prismatic_shell_to_beam] and wind turbine blade modeling [@opensg_timoshenko]. These applications highlight the potential of OpenSG to enable accurate, scalable, and open-source solutions for structural modeling challenges across wind energy and beyond.

# Acknowledgements
OpenSG has been developed within the Holistic, Multi-Fidelity Wind Farm Design Optimization and Model Coordination project under the U.S. Department of Energy Wind Energy Technologies Office. It is currently being evaluated for its suitability to transform wind turbine blade design, mainly due to its ability to incorporate high-fidelity structural details into efficient beam models.

Sandia National Laboratories is a multimission laboratory managed and operated by National Technology and Engineering Solutions of Sandia, LLC., a wholly owned subsidiary of Honeywell International, Inc., for the U.S. Department of Energy’s National Nuclear Security Administration under contract DE-NA-0003525.

# References 
