from functools import partial

import numpy as np
import jax.lax as lax
import jax.numpy as jnp

from jax import device_put
from jax._src.scipy.sparse.linalg import (
    _identity,
    _vdot_real_tree,
    _add,
    _sub,
    _mul,
    _normalize_matvec,
    _shapes,
)
from jax.tree_util import tree_leaves, tree_map, tree_structure, tree_reduce, Partial


def _cg_solve(
    A, b, x0=None, *, maxiter, tol=1e-5, atol=0.0, M=_identity, return_diagnostics=False
):

    # tolerance handling uses the "non-legacy" behavior of scipy.sparse.linalg.cg
    bs = _vdot_real_tree(b, b)
    atol2 = jnp.maximum(jnp.square(tol) * bs, jnp.square(atol))

    # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method

    def cond_fun(value):
        _, r, gamma, _, k, _ = value
        rs = gamma.real if M is _identity else _vdot_real_tree(r, r)
        return (rs > atol2) & (k < maxiter)

    def body_fun(value):
        x, r, gamma, p, k, r_norm_ks = value
        Ap = A(p)
        alpha = gamma / _vdot_real_tree(p, Ap).astype(dtype)
        x_ = _add(x, _mul(alpha, p))
        r_ = _sub(r, _mul(alpha, Ap))
        z_ = M(r_)
        gamma_ = _vdot_real_tree(r_, z_).astype(dtype)
        beta_ = gamma_ / gamma
        p_ = _add(z_, _mul(beta_, p))
        return x_, r_, gamma_, p_, k + 1, r_norm_ks.at[k + 1].set(jnp.linalg.norm(r_))

    r0 = _sub(b, A(x0))
    p0 = z0 = M(r0)
    dtype = jnp.result_type(*tree_leaves(p0))
    gamma0 = _vdot_real_tree(r0, z0).astype(dtype)
    r_norm_ks0 = jnp.zeros((maxiter,)).at[0].set(jnp.linalg.norm(r0))
    initial_value = (x0, r0, gamma0, p0, 0, r_norm_ks0)

    # Keith: small modification here to unpack the residual and iteration count, which are
    # returned in a dict
    x_final, _, _, _, k_final, r_norm_ks_final = lax.while_loop(
        cond_fun, body_fun, initial_value
    )

    return x_final, {
        "iterations": k_final,
        "residual_norm_history": r_norm_ks_final,
    }


def _isolve(
    _isolve_solve,
    A,
    b,
    x0=None,
    *,
    tol=1e-5,
    atol=0.0,
    maxiter=None,
    M=None,
    check_symmetric=False,
):
    if x0 is None:
        x0 = tree_map(jnp.zeros_like, b)

    b, x0 = device_put((b, x0))

    if maxiter is None:
        size = sum(bi.size for bi in tree_leaves(b))
        maxiter = 10 * size  # copied from scipy

    if M is None:
        M = _identity
    A = _normalize_matvec(A)
    M = _normalize_matvec(M)

    if tree_structure(x0) != tree_structure(b):
        raise ValueError(
            "x0 and b must have matching tree structure: "
            f"{tree_structure(x0)} vs {tree_structure(b)}"
        )

    if _shapes(x0) != _shapes(b):
        raise ValueError(
            "arrays in x0 and b must have matching shapes: "
            f"{_shapes(x0)} vs {_shapes(b)}"
        )

    isolve_solve = partial(
        _isolve_solve, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M
    )

    # real-valued positive-definite linear operators are symmetric
    def real_valued(x):
        return not issubclass(x.dtype.type, np.complexfloating)

    symmetric = all(map(real_valued, tree_leaves(b))) if check_symmetric else False
    # Keith: very small modification here to return 'info' and set hax_aux to True
    x, info = lax.custom_linear_solve(
        A,
        b,
        solve=isolve_solve,
        transpose_solve=isolve_solve,
        symmetric=symmetric,
        has_aux=True,
    )
    return x, info


# Keith: verbatim of jax's cg, but using my modified functions
def cg(A, b, x0=None, *, tol=1e-5, atol=0.0, maxiter=None, M=None):
    return _isolve(
        _cg_solve,
        A=A,
        b=b,
        x0=x0,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        M=M,
        check_symmetric=True,
    )
