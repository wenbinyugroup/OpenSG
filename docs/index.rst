.. _home:

OpenSG
======

OpenSG is an open source platform for multiscale structural analysis using FEniCS backend which works on the Mechanics of Structure Genome (MSG) 
theory. OpenSG has capabilties to obtain the structural homogenized properties in terms of either of Cauchy continuum (3D) model, 
plate/shell model or beam model. The obtained homogenized structural properties are accurate, fast and precise for complex structural 
profile like wind blades. 

OpenSG evaluates the homogenization using the mathematical building block of structure, called Structure Gene (SG), 
which can be of one, two or three dimension. It has the capability to generate local stress and displacement field 
(dehomogenization) of the structure based on user defined input load. OpenSG gives the research community to 
leverage the advantages of MSG theory for generating complex models based on the specific applications. 

Key Features
------------

- **Shell Element Analysis**: Support for shell-based blade modeling with composite layup definitions
- **Solid Element Analysis**: 3D volumetric analysis capabilities for detailed structural assessment
- **Timoshenko Beam Theory**: Implementation of advanced beam theory for accurate blade modeling
- **Composite Material Support**: Comprehensive material property definitions for wind blade composites
- **Segment-Based Workflow**: Ability to work with individual blade segments
- **Finite Element Integration**: Built on DOLFINx for robust finite element analysis

Applications
------------

OpenSG is designed for:

- Wind turbine blade structural design
- Composite material analysis
- Blade optimization studies
- Research and development in wind energy
- Educational purposes in structural engineering


.. TODO
    - Motivation for OpenSG
    - Novelty of OpenSG
    - Capabilities of OpenSG

.. toctree::
   :maxdepth: 1
   :hidden:

   Home <self>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Introduction

   installation
   getting-started
   examples

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Usage
   
   framework
   input
   theory

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Backmatter

   citing_opensg
   contributing
   ../license
   glossary
   reference 

.. toctree::
   :maxdepth: 1
   :hidden:
   
   api-doc

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
