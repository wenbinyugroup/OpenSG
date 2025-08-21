---
title: 'OpenSG: Open Source Structural Analysis for Wind Turbine Blades'
tags:
  - Python
  - wind energy
  - structural analysis
  - finite element method
  - composite materials
  - blade design
authors:
  - name: Kirk Bonney
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: 1
  - name: Akshat Bagla
    orcid: 0000-0000-0000-0000
    affiliation: 2
  - name: Ernesto Camarena
    orcid: 0000-0000-0000-0000
    affiliation: 1

affiliations:
 - name: Sandia National Laboratories, Albuquerque, NM, USA
   index: 1
 - name: Purdue University, West Lafayette, IN, USA
   index: 2

date: 21 August 2025
bibliography: paper.bib

---

# JOSS paper

JOSS welcomes submissions from broadly diverse research areas. For this reason, we require that authors include in the paper some sentences that explain the software functionality and domain of use to a non-specialist reader. We also require that authors explain the research applications of the software. The paper should be between 250-1000 words. Authors submitting papers significantly longer than 1000 words may be asked to reduce the length of their paper.

# Summary

A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience.

(README blurb)

OpenSG is an open source platform for multiscale structural analysis using FEniCS backend which works on the Mechanics of Structure Genome (MSG) 
theory. OpenSG has capabilties to obtain the structural homogenized properties in terms of either of Cauchy continuum (3D) model, plate/shell model or beam model. The obtained homogenized structural properties are accurate, fast and precise for complex structural profile like wind blades. 

OpenSG evaluates the homogenization using the mathematical building block of structure, called Structure Gene (SG), which can be of one, two or three dimension. It has the capability to generate local stress and displacement field (dehomogenization) of the structure based on user defined input load. OpenSG gives the research community to leverage the advantages of MSG theory for generating complex models based on the specific applications. 

# Statement of need

A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work.

TODO (Ernesto)

# Other

Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it.

Any other relevant discussion we want to put here. Could include brief description of deeper theory here.

# Acknowledgements
Acknowledgement of any financial support.

# References 