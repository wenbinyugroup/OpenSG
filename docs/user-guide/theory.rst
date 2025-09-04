.. _theory:

Theory
======

OpenSG platform is the numerical implementation of a unified and revolutionary multiscale structure mechanics theory, Mechanics of Structure Genome (MSG), which provides a rigorous and systematic approach to modeling advanced structures featuring general anisotropy and heterogeneity, including beams, plates, shells, and 3D structures. OpenSG uses Structure Gene (SG) as the mathematical building block to define heterogeneity and anisotropy of the structure like, laminated composite plates with 1D SG (heterogeneity in one direction), prismatic beams with 2D SG, and aperiodic beams with 3D SG for defining the mathematical unit (SG). 

Beam Structures
---------------

For beam like structures such as aircraft wing, helicopter blades, and wind turbine blades, MSG uses Variational Asymptotic Method (VAM) to split the original 3D FEA problem into 2D cross-sectional analysis and 1D beam analysis. OpenSG performs the 2D cross-sectional analysis to predict Timoshenko beam stiffness in two separate ways: 

1. 2D SG (VABS Equivalent) - with input of material elastic properties (Original model: 3D Elasticity) 

2. 1D SG (PreComp Equivalent) - with input of material layup (Original model: 2D Shell) 

To include 3D curvature effects and estimate the local buckling, OpenSG incorporates the 3D SG for both solid (original model: 3D Elasticity) and shell mesh elements (original model: 2D Shell) to predict beam properties of realistic tapered blade segment. The aperiodic boundaries are constrained using known fluctuating displacements at the respective boundaries (mapped from 3D SG). 

For shell structures, OpenSG computes two-step homogenization including a) Classical stiffness from known layups, b) Timoshenko beam stiffness (using obtained elemental classical plate stiffness for shell strain energy minimization). 

Local Material Orientation
--------------------------

OpenSG takes local coordinate reference frames using elemental direction cosine matrix provided in input .yaml file, where the direction cosine matrix considers all three Euler angle rotations in local elemental frame. The global beam reference frame :math:`xyz` is inline with BeamDyn coordinate system having z-axis along the beam reference line. However, the local elemental reference frame :math:`123` is aligned such that the normal direction :math:`e_3` is inward pointing to the beam reference line. 

For shell structures, the local curvilinear frame :math:`123` considers the normal to shell element :math:`e_3` is in inward direction, :math:`e_1` points along the global beam axis and :math:`e_2` follows the right-hand rule. OpenSG considers the outer-mold layer (OML) as the shell-reference surface, and the layup proceeds along the positive :math:`e_3` direction. In other words, the origin of local frame lies on the outer reference surface like over 1D SG (line elements) and layup proceeds along the positive :math:`e_3` direction. 

For aperiodic 3D SG, the boundaries are mapped geometrically from 3D SG to perform MSG separately for obtaining unknown fluctuating displacements at boundary nodes. For instance, 3D SG (shell elements) would map 1D SG (line elements) boundary meshes at both aperiodic ends transferring the corresponding material layup and orientation information. This operation is automatically taken care by OpenSG and user gets the advantage to have Timoshenko beam stiffness at boundaries in addition to that of 3D SG. 

Dehomogenization
----------------

OpenSG performs dehomogenization (local stress recovery) from input beam reaction forces which gives 

.. math::

   \sigma^{3D}=[\sigma_{11}, \sigma_{22},\sigma_{33},\sigma_{23},\sigma_{13},\sigma_{12} ]

for original 3D elastic model (solid elements), and similarly, 

.. math::

   N=[ N_{11}, N_{22},N_{12},M_{11},M_{22},M_{12} ]

for original 2D shell model (shell elements). These local stresses can be predicted (like VABS) over cross-section for respective :math:`\sigma^{3D}` over 2D SG (solid elements) and :math:`N^{2D}` over 1D SG (shell elements). 

Local Buckling
----------------

OpenSG predicts the local buckling through solving generalized eigensolver, 

.. math::

   K X=\lambda K_G X
.. math::

   K_G X =\frac{1}{\lambda} K X

where, :math:`K` represents the elastic stiffness matrix and :math:`K_G` the prestressed matrix formed using recovered local stress field. :math:`\frac{1}{\lambda_{max}}` would give the minimum positive eigenvalue (buckling load), and corresponding eigen vector, X represents the local buckling mode shapes. 

Plate and 3D model
------------------

For Plate structures, like functionally graded, ply-drop structures, MSG splits the original 3D FEA problem into 1D composite laminate homogenization and 2D plate structural analysis. OpenSG performs the Kirchhoff-Love or Reissner-Mindlin like constitutive plate properties whereas the corresponding plate structural analysis can be performed in Abaqus, Ansys or Nastran. 

Similarly, For 3D heterogeneous structures, like metamaterials, OpenSG perform the general homogenization in a better way than traditional RVE or MHT approach using 3D (solid elements) based unit cell (SG). However, MSG doesn't require the boundary to be periodic, for instance, the uni-directional fiber laminate requires 2D SG to output the same 9 elastic properties :math:`E_1, E_2,E_3, G_{12}, G_{13}, G_{23}, \nu_{12},\nu_{13},\nu_{23}`, refer []. 
