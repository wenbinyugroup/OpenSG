"""PETSc method options for the GAMG solver route (sg_gamg).

Vendored from the jetsci layer of fea-in-jax (Keith Ballard, Air Force Research
Laboratory; Distribution Statement A, case AFRL-2025-4644), reduced to the pure
option surface that OpenSG uses: the Mat, KSP and PC type names that sg_gamg hands
to petsc4py.  The GPU lifecycle layers of jetsci (CuPy buffer callbacks) are not
part of OpenSG.

    o = PETScMethodOptions(mat_type="aij", ksp_type="cg", pc_type="gamg")
    o.matrix_construction_options()   # ("aij",)
    o.ksp_construction_options()      # ("cg",)
    o.pc_construction_options()       # ("gamg",)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class PETScMatrixType(Enum):
    AIJ = auto()
    AIJCUSPARSE = auto()
    MPIAIJCUSPARSE = auto()


class PETScKSPType(Enum):
    CG = auto()
    GMRES = auto()
    LGMRES = auto()
    BICGSTAB = auto()
    MINRES = auto()


class PETScPCType(Enum):
    NONE = auto()
    JACOBI = auto()
    ILU = auto()
    GAMG = auto()


_MATRIX_NAMES = {
    PETScMatrixType.AIJ: "aij",
    PETScMatrixType.AIJCUSPARSE: "aijcusparse",
    PETScMatrixType.MPIAIJCUSPARSE: "mpiaijcusparse",
}

_KSP_NAMES = {
    PETScKSPType.CG: "cg",
    PETScKSPType.GMRES: "gmres",
    PETScKSPType.LGMRES: "lgmres",
    PETScKSPType.BICGSTAB: "bcgs",
    PETScKSPType.MINRES: "minres",
}

_PC_NAMES = {
    PETScPCType.NONE: "none",
    PETScPCType.JACOBI: "jacobi",
    PETScPCType.ILU: "ilu",
    PETScPCType.GAMG: "gamg",
}


def _coerce_enum(value, enum_type, names):
    """An enum member from the member itself, its name or its PETSc type name.

    In:  value Enum | str; enum_type Enum class; names dict Enum -> str
    Out: Enum member (ValueError when nothing matches)."""
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        upper = value.upper()
        if upper in enum_type.__members__:
            return enum_type[upper]
        for enum_value, petsc_name in names.items():
            if value.lower() == petsc_name:
                return enum_value
    raise ValueError(f"Expected {enum_type.__name__}, got {value!r}")


@dataclass(frozen=True)
class PETScMethodOptions:
    """Mat, KSP and PC type selection, accepted as enum members or PETSc names.

    In:  mat_type PETScMatrixType | str (default aijcusparse); ksp_type PETScKSPType | str
         (default cg); pc_type PETScPCType | str (default ilu)
    Out: the *_construction_options() methods return 1-tuples of PETSc type names."""

    mat_type: PETScMatrixType = PETScMatrixType.AIJCUSPARSE
    ksp_type: PETScKSPType = PETScKSPType.CG
    pc_type: PETScPCType = PETScPCType.ILU

    def __post_init__(self):
        object.__setattr__(self, "mat_type", _coerce_enum(self.mat_type, PETScMatrixType, _MATRIX_NAMES))
        object.__setattr__(self, "ksp_type", _coerce_enum(self.ksp_type, PETScKSPType, _KSP_NAMES))
        object.__setattr__(self, "pc_type", _coerce_enum(self.pc_type, PETScPCType, _PC_NAMES))

    def matrix_construction_options(self):
        return (_MATRIX_NAMES[self.mat_type],)

    def ksp_construction_options(self):
        return (_KSP_NAMES[self.ksp_type],)

    def pc_construction_options(self):
        return (_PC_NAMES[self.pc_type],)
