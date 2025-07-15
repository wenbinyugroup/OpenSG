"""Default configuration parameters for OpenSG."""

# Solver settings
SOLVER_SETTINGS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_solver_type": "mumps",
    "pc_factor_setup_solver_type": "mumps",
    "pc_factor_mumps_icntl_24": 1,  # detect null pivots
    "pc_factor_mumps_icntl_25": 0,  # do not compute null space again
}

# Mesh generation settings
MESH_SETTINGS = {
    "quadrature_degree": 4,
    "mesh_format": "gmsh",
    "mesh_version": "2.2",
}

# Material properties (example defaults)
DEFAULT_MATERIALS = {
    "carbon_fiber": {
        "E": [130000.0, 10000.0, 10000.0],  # MPa
        "G": [5000.0, 5000.0, 3500.0],      # MPa
        "nu": [0.28, 0.28, 0.28],           # Poisson's ratios
        "density": 1600.0,                   # kg/m³
    },
    "glass_fiber": {
        "E": [45000.0, 10000.0, 10000.0],   # MPa
        "G": [5000.0, 5000.0, 3500.0],      # MPa
        "nu": [0.25, 0.25, 0.25],           # Poisson's ratios
        "density": 1800.0,                   # kg/m³
    },
    "foam": {
        "E": [50.0, 50.0, 50.0],            # MPa
        "G": [20.0, 20.0, 20.0],            # MPa
        "nu": [0.3, 0.3, 0.3],              # Poisson's ratios
        "density": 100.0,                    # kg/m³
    }
}

# Analysis settings
ANALYSIS_SETTINGS = {
    "beam_theory": "timoshenko",  # "euler_bernoulli" or "timoshenko"
    "include_shear": True,
    "include_rotary_inertia": True,
    "tolerance": 1e-6,
    "max_iterations": 1000,
}

# Output settings
OUTPUT_SETTINGS = {
    "save_mesh": True,
    "save_stiffness_matrices": True,
    "save_displacement_fields": False,
    "output_format": "numpy",  # "numpy", "matlab", "csv"
    "precision": 6,
} 