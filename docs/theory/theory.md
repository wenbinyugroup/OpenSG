# Theory and Implementation Details

## Nonlinear Solver

To derive Newton's method, it is convenient to start with a Taylor series expansion of the residual function:

$$ R(u_i) = R(u_{i-1}) + \frac{\partial \textbf{R}}{\partial \textbf{u}} \Delta \textbf{u} + O(||\Delta \textbf{u}||^2) $$

We want to solve for $ R(u_i) = 0 $, so rearrange the equations to solve for $\Delta \textbf{u}$:

$$ \frac{\partial \textbf{R}}{\partial \textbf{u}} \Delta \textbf{u} \approx -R(u_{i-1}) $$

By iterating, $u_i$ will quadratically converge given the $R(u)$ meets certain conditions.

To enforce Dirichlet boundary conditions (BCs), we typically use in-place elimination, which effectively eliminates the constrained degrees of freedom (DoFs) while leaving the respective rows / columns in the system of equations. Let $\textbf{U}_i$ be a vector that is the values of the Dirichlet BCs where applicable and 0 elsewhere. The Jacobian, $\frac{\partial \textbf{R}}{\partial \textbf{u}}$, can be modified to have 0's in the rows / columns and 1 on the diagonal for the constrained DoFs, but the RHS must be modified as well.

If we let $\textbf{D}_i$ be the indices of the Dirichlet DoFs, then the incremental solution for the Dirichlet BCs can be calculated

$$ [\Delta \textbf{U}_i]_j = 
    \{\begin{array}{lr} 
        [\textbf{U}_i]_j - [\textbf{u}_{i-1}]_j, & \text{if } j \in \textbf{D}_i] \\
        0, & \text{otherwise}
    \end{array}\} $$

The adjusted system of equations that is typically solved becomes

$$ \frac{\partial \textbf{R}}{\partial \textbf{u}} \Delta \textbf{u} = -R(u_{i-1}) - \frac{\partial \textbf{R}}{\partial \textbf{u}} \Delta \textbf{U}_i $$

However, in `calculate_residual_w_dirichlet`, the residual calculated is:

$$ \textbf{R} =  \frac{\partial \textbf{R}}{\partial \textbf{u}} \textbf{v} $$

where $\textbf{v}$ is adjusted to include the Dirichlet BCs. Importantly, this means, that $\textbf{R}$ already incorporates the RHS term

$$ - \frac{\partial \textbf{R}}{\partial \textbf{u}} \Delta \textbf{U}_i $$

## Variables Indicating Dimensions of Arrays
Superscripts will be used to denote the rank, since within the programming implementation the shape of arrays will not include the rank index. The rank index is useful to discuss the distributed algorithm but does not affect the stored quantities.
* $\mathcal{R}$: total # of MPI ranks
* $\mathcal{B}^i$: # of batches of elements used for computations on the $i^\mathsf{th}$ MPI rank
* $\mathcal{V}^i$: # of nodes used for computations on the $i^\mathsf{th}$ MPI rank
* $\mathcal{E}^i_j$: # of elements in the $j^\mathsf{th}$ batch on the $i^\mathsf{th}$ rank
* $\mathcal{D}$: # of dimensions in the global coordinate system, which is also # components for displacement
* $\mathcal{I}^i_j$: # of dimensions in the isoparametric coordinate system for the $j^\mathsf{th}$ batch of elements, on the $i^\mathsf{th}$ rank. (should match $\mathcal{D}$ for solid elements)
* $\mathcal{N}^i_j$: # of nodes for each element in the $j^\mathsf{th}$ batch of elements on the $i^\mathsf{th}$ rank.
* $\mathcal{Q}^i_j$: # of quadrature points in each element for the $j^\mathsf{th}$ batch of elements on the $i^\mathsf{th}$ rank.
* $\mathcal{M}^i_j$: # of material parameters in the constitutive model for the $j^\mathsf{th}$ batch of elements on the $i^\mathsf{th}$ rank.
* $\mathcal{S}$: # of strain components (generally determined by $\mathcal{D}$ and/or $\mathcal{U}$)
* $\mathcal{U}$: # of components of the solution per basis function
* $\mathcal{F}^i$: total # of degrees of freedom on the $i^\mathsf{th}$ rank. 
* $\mathcal{P}^i$: # patches on MPI rank $i$
* $\mathcal{K}^i_{j}$: # of vertices on patch $j$ on MPI rank $i$
* $\mathcal{L}^i_{j}$: # of elements on patch $j$ on MPI rank $i$
* $\mathcal{G}^i_{j}$: # of degrees of freedom on patch $j$ on MPI rank $i$
