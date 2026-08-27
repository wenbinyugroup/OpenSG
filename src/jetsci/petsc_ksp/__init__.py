"""Level-3 PETSc method API.

This is the intended import surface for manual PETSc lifecycle control.
Do not import from `v10.JaxCallsPETSc._primitives` unless you are editing
primitive rules or raw callback implementations.
"""

from .options import PETScKSPType, PETScMatrixType, PETScMethodOptions, PETScPCType

# the lifecycle layers below are cupy/CUDA-bound (buffer callbacks on
# GPU memory); a CPU-only install keeps the pure options surface above
try:
    from .linear_methods import (
        COOData,
        KSP_solve,
        KSP_solve_from_COOData,
        KSP_solve_from_coo_data,
        KSP_solve_transpose,
        cleanup_ksp,
        cleanup_matrix,
        cleanup_pc,
        evaluate_matrix_function,
        to_COOData_object,
        init_ksp,
        init_matrix_from_coo,
        init_matrix_from_COOData,
        init_matrix_from_function,
        init_pc,
        solve_ksp,
        solve_ksp_from_coo_data,
        solve_ksp_transpose,
        solve_once,
        update_matrix_values,
    )
    from .solver_call_methods import (
    PETScLinearSolverObjects,
    buildSolverObjects,
    buildSolverObjectsFromCOOData,
    build_solver_objects,
    build_solver_objects_from_coo_data,
    cleanupSolverObjects,
    cleanup_solver_objects,
    runSimulationWithSolverObjects,
    run_simulation_with_solver_objects,
        solveWithSolverObjects,
        solve_with_solver_objects,
    )
except ImportError:
    pass

__all__ = [
    "COOData",
    "KSP_solve",
    "KSP_solve_from_COOData",
    "KSP_solve_from_coo_data",
    "KSP_solve_transpose",
    "PETScKSPType",
    "PETScLinearSolverObjects",
    "PETScMatrixType",
    "PETScMethodOptions",
    "PETScPCType",
    "buildSolverObjects",
    "buildSolverObjectsFromCOOData",
    "build_solver_objects",
    "build_solver_objects_from_coo_data",
    "cleanup_ksp",
    "cleanup_matrix",
    "cleanup_pc",
    "cleanupSolverObjects",
    "cleanup_solver_objects",
    "evaluate_matrix_function",
    "to_COOData_object",
    "init_ksp",
    "init_matrix_from_coo",
    "init_matrix_from_COOData",
    "init_matrix_from_function",
    "init_pc",
    "runSimulationWithSolverObjects",
    "run_simulation_with_solver_objects",
    "solveWithSolverObjects",
    "solve_ksp",
    "solve_ksp_from_coo_data",
    "solve_ksp_transpose",
    "solve_with_solver_objects",
    "solve_once",
    "update_matrix_values",
]
