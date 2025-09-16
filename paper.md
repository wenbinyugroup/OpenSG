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

OpenSG is an open-source platform for multiscale structural mechanics using FEniCS backend which implements the Mechanics of Structure Genome (MSG) theory [@yu2019]. This unified and revolutionary theory provides a rigorous and systematic approach to modeling advanced structures featuring general anisotropy and heterogeneity, including beams, plates, shells, and continuum structures.

Being a micromechanics code, OpenSG is useful for incorporating micro- and mesoscale features into other analysis programs via homogenization and dehomogenization. Homogenization is the process of obtaining homogenized structural properties whereas, dehomogenization is recovering the local stress and displacement fields. The homogenization and dehomogenization take place over the user-defined domain called the Structure Gene (SG). The SG serves as the mathematical building block to define the heterogeneity and anisotropy. It can be meshed with either 1D line elements, 2D quadrilateral or triangular elements, or 3D hexahedron or tetrahedron elements. The SG can then undergo homogenization to obtain structural properties in terms of Cauchy continuum model, plate/shell model, or beam model, providing accurate, fast, and precise analysis for complex structural profiles like wind turbine blades. Thus, enabling the research community to leverage the advantages of MSG theory for generating complex models for specific applications. 

# Statement of need

The MSG theory was first implemented in SwiftComp [@swiftcomp], a commercial code. It offers unification of the following codes: VAMUCH for unit cells of materials [@vamuch], VAPAS [@vapas] for plates and shells, and VABS [@cesnik1997vabs] for beams. In addition to replicating the core of SwiftComp's capabilities [@opensg], OpenSG extends the MSG theory to allow for aperiodic beam SGs made of either shell [@opensg_prismatic_shell_to_beam; @opensg_RM_shell] or sold [@opensg_tapered_solid] elements. This is particularly useful for modeling nonprismatic structures and for accounting for 3D phenomenon, such as panel buckling. Some examples of nonprismatic structures are airplane wings, propellers, offshore jacketed structures, wind blades, tapered tubes and rods, non prismatic bridges. 


# Theoretical Framework

TODO (Akshat) 

NOTE: As a placeholder, I pulled in the text from `theory.rst`, but this should be rewritten to be a simple summary and avoid details. JOSS papers typically don't include extensive theoretical content.

## Beam Structures

For beam-like structures such as aircraft wings, helicopter blades, and wind turbine blades, MSG uses the Variational Asymptotic Method (VAM) to split the original 3D finite element analysis problem into 2D cross-sectional analysis and 1D beam analysis. OpenSG performs the 2D cross-sectional analysis to predict Timoshenko beam stiffness in two separate ways:

1. **2D SG (VABS Equivalent)** - with input of material elastic properties (Original model: 3D Elasticity)
2. **1D SG (PreComp Equivalent)** - with input of material layup (Original model: 2D Shell)

To include 3D curvature effects and estimate local buckling, OpenSG incorporates the 3D SG for both solid (original model: 3D Elasticity) and shell mesh elements (original model: 2D Shell) to predict beam properties of realistic tapered blade segments. The aperiodic boundaries are constrained using known fluctuating displacements at the respective boundaries (mapped from 3D SG).

For shell structures, OpenSG computes two-step homogenization including: (a) Classical stiffness from known layups, and (b) Timoshenko beam stiffness using obtained elemental classical plate stiffness for shell strain energy minimization.

## Local Material Orientation

OpenSG uses local coordinate reference frames with elemental direction cosine matrices provided in input YAML files, considering all three Euler angle rotations in the local elemental frame. The global beam reference frame *xyz* aligns with the BeamDyn coordinate system having the z-axis along the beam reference line. The local elemental reference frame *123* is aligned such that the normal direction *e₃* points inward toward the beam reference line.

For shell structures, the local curvilinear frame *123* considers the normal to shell element *e₃* in the inward direction, *e₁* points along the global beam axis, and *e₂* follows the right-hand rule. OpenSG considers the outer-mold-line (OML) as the shell-reference surface, with the layup proceeding along the positive *e₃* direction.

For aperiodic 3D SG, the boundaries are mapped geometrically from 3D SG to perform MSG separately for obtaining unknown fluctuating displacements at boundary nodes. For instance, 3D SG (shell elements) would map 1D SG (line elements) boundary meshes at both aperiodic ends transferring the corresponding material layup and orientation information. This operation is automatically taken care by OpenSG and user gets the advantage to have Timoshenko beam stiffness at boundaries in addition to that of 3D SG.

## Dehomogenization and Local Stress Recovery

OpenSG performs dehomogenization (local stress recovery) from input beam reaction forces, providing:

- σ³ᴰ = [σ₁₁, σ₂₂, σ₃₃, σ₂₃, σ₁₃, σ₁₂] for original 3D elastic model (solid elements)
- N = [N₁₁, N₂₂, N₁₂, M₁₁, M₂₂, M₁₂] for original 2D shell model (shell elements)

These local stresses can be predicted over the cross-section for respective σ³ᴰ over 2D SG (solid elements) and N²ᴰ over 1D SG (shell elements).

## Local Buckling Analysis

OpenSG predicts local buckling through solving a generalized eigenvalue problem:

KX = λK_GX

where K represents the elastic stiffness matrix and K_G the prestressed matrix formed using the recovered local stress field. The minimum positive eigenvalue (1/λ_max) gives the buckling load, and the corresponding eigenvector X represents the local buckling mode shapes.

## Plate and 3D Structures

For plate structures, such as functionally graded and ply-drop structures, MSG splits the original 3D finite element analysis problem into 1D composite laminate homogenization and 2D plate structural analysis. OpenSG performs Kirchhoff-Love or Reissner-Mindlin constitutive plate properties, while the corresponding plate structural analysis can be performed in Abaqus, Ansys, or Nastran.

For 3D heterogeneous structures, such as metamaterials, OpenSG performs general homogenization more effectively than traditional Representative Volume Element (RVE) or Mean Field Theory (MFT) approaches using 3D (solid elements) based unit cells (SG). MSG does not require periodic boundaries; for instance, unidirectional fiber laminates require 2D SG to output the same 9 elastic properties: E₁, E₂, E₃, G₁₂, G₁₃, G₂₃, ν₁₂, ν₁₃, ν₂₃.

# Acknowledgements
TODO Acknowledgement of any financial support.

# References 


