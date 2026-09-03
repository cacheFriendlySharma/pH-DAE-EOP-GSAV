from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from linear_solvers import factorize,solve_factorized


VectorFunction=Callable[[np.ndarray],np.ndarray]
MatrixFunction=Callable[[np.ndarray],Any]


@dataclass
class NewtonResult:
    x: np.ndarray
    converged: bool
    iterations: int
    residual_norm: float
    residual_evaluations: int
    jacobian_evaluations: int
    factorizations: int
    linear_solves: int


def _validate_newton_parameters(tol,max_iter):
    if tol<=0.0:
        raise ValueError("tol must be positive.")

    if max_iter<0:
        raise ValueError("max_iter must be nonnegative.")


def _evaluate_residual(
    residual: VectorFunction,
    x: np.ndarray,
):
    F=np.asarray(
        residual(x),
        dtype=float,
    ).ravel()

    if F.shape!=x.shape:
        raise ValueError(
            f"Residual must have shape {x.shape}, "
            f"received {F.shape}."
        )

    residual_norm=float(
        np.linalg.norm(
            F,
            ord=np.inf,
        )
    )

    return F,residual_norm


# ============================================================
# Full Newton
# ============================================================

def full_newton(
    residual: VectorFunction,
    jacobian: MatrixFunction,
    x_0: np.ndarray,
    tol: float=1e-10,
    max_iter: int=20,
) -> NewtonResult:
    """
    Full Newton iteration

        J(x_j) delta_j = -F(x_j),
        x_{j+1} = x_j + delta_j,

    with a new Jacobian evaluation and factorization at every
    nonlinear iteration.

    Convergence criterion:

        ||F(x_j)||_inf <= tol.
    """
    _validate_newton_parameters(
        tol,
        max_iter,
    )

    x=np.asarray(
        x_0,
        dtype=float,
    ).copy()

    if x.ndim!=1 or x.size==0:
        raise ValueError(
            "x_0 must be a nonempty one-dimensional array."
        )

    residual_evaluations=0
    jacobian_evaluations=0
    factorizations=0
    linear_solves=0

    for iteration in range(max_iter+1):

        F,residual_norm=_evaluate_residual(
            residual,
            x,
        )

        residual_evaluations+=1

        # ----------------------------------------------------
        # Non-finite nonlinear iterate
        # ----------------------------------------------------

        if not np.isfinite(residual_norm):
            return NewtonResult(
                x=x,
                converged=False,
                iterations=iteration,
                residual_norm=np.inf,
                residual_evaluations=residual_evaluations,
                jacobian_evaluations=jacobian_evaluations,
                factorizations=factorizations,
                linear_solves=linear_solves,
            )

        # ----------------------------------------------------
        # Convergence
        # ----------------------------------------------------

        if residual_norm<=tol:
            return NewtonResult(
                x=x,
                converged=True,
                iterations=iteration,
                residual_norm=residual_norm,
                residual_evaluations=residual_evaluations,
                jacobian_evaluations=jacobian_evaluations,
                factorizations=factorizations,
                linear_solves=linear_solves,
            )

        if iteration==max_iter:
            break

        # ----------------------------------------------------
        # Current Jacobian
        # ----------------------------------------------------

        J=jacobian(x)
        jacobian_evaluations+=1

        J_lu=factorize(J)
        factorizations+=1

        # ----------------------------------------------------
        # Newton correction
        # ----------------------------------------------------

        delta=np.asarray(
            solve_factorized(
                J_lu,
                -F,
            ),
            dtype=float,
        ).ravel()

        linear_solves+=1

        if delta.shape!=x.shape:
            raise ValueError(
                f"Newton correction must have shape {x.shape}, "
                f"received {delta.shape}."
            )

        if not np.all(np.isfinite(delta)):
            return NewtonResult(
                x=x,
                converged=False,
                iterations=iteration,
                residual_norm=residual_norm,
                residual_evaluations=residual_evaluations,
                jacobian_evaluations=jacobian_evaluations,
                factorizations=factorizations,
                linear_solves=linear_solves,
            )

        x+=delta

    return NewtonResult(
        x=x,
        converged=False,
        iterations=max_iter,
        residual_norm=residual_norm,
        residual_evaluations=residual_evaluations,
        jacobian_evaluations=jacobian_evaluations,
        factorizations=factorizations,
        linear_solves=linear_solves,
    )


# ============================================================
# Modified Newton
# ============================================================

def modified_newton(
    residual: VectorFunction,
    jacobian: MatrixFunction,
    x_0: np.ndarray,
    tol: float=1e-10,
    max_iter: int=20,
) -> NewtonResult:
    """
    Modified Newton iteration.

    The Jacobian is evaluated and factorized once at x_0:

        J_0 = J(x_0),

    and the same factorization is reused for every nonlinear
    correction within the solve:

        J_0 delta_j = -F(x_j),
        x_{j+1} = x_j + delta_j.

    Convergence criterion:

        ||F(x_j)||_inf <= tol.
    """
    _validate_newton_parameters(
        tol,
        max_iter,
    )

    x=np.asarray(
        x_0,
        dtype=float,
    ).copy()

    if x.ndim!=1 or x.size==0:
        raise ValueError(
            "x_0 must be a nonempty one-dimensional array."
        )

    residual_evaluations=0
    jacobian_evaluations=0
    factorizations=0
    linear_solves=0

    # --------------------------------------------------------
    # Initial residual
    # --------------------------------------------------------

    F,residual_norm=_evaluate_residual(
        residual,
        x,
    )

    residual_evaluations+=1

    if not np.isfinite(residual_norm):
        return NewtonResult(
            x=x,
            converged=False,
            iterations=0,
            residual_norm=np.inf,
            residual_evaluations=residual_evaluations,
            jacobian_evaluations=jacobian_evaluations,
            factorizations=factorizations,
            linear_solves=linear_solves,
        )

    if residual_norm<=tol:
        return NewtonResult(
            x=x,
            converged=True,
            iterations=0,
            residual_norm=residual_norm,
            residual_evaluations=residual_evaluations,
            jacobian_evaluations=jacobian_evaluations,
            factorizations=factorizations,
            linear_solves=linear_solves,
        )

    # No nonlinear update is allowed.
    if max_iter==0:
        return NewtonResult(
            x=x,
            converged=False,
            iterations=0,
            residual_norm=residual_norm,
            residual_evaluations=residual_evaluations,
            jacobian_evaluations=jacobian_evaluations,
            factorizations=factorizations,
            linear_solves=linear_solves,
        )

    # --------------------------------------------------------
    # Frozen Jacobian
    # --------------------------------------------------------

    J=jacobian(x)
    jacobian_evaluations+=1

    J_lu=factorize(J)
    factorizations+=1

    # --------------------------------------------------------
    # Modified Newton iterations
    # --------------------------------------------------------

    for iteration in range(1,max_iter+1):

        delta=np.asarray(
            solve_factorized(
                J_lu,
                -F,
            ),
            dtype=float,
        ).ravel()

        linear_solves+=1

        if delta.shape!=x.shape:
            raise ValueError(
                f"Newton correction must have shape {x.shape}, "
                f"received {delta.shape}."
            )

        if not np.all(np.isfinite(delta)):
            return NewtonResult(
                x=x,
                converged=False,
                iterations=iteration-1,
                residual_norm=residual_norm,
                residual_evaluations=residual_evaluations,
                jacobian_evaluations=jacobian_evaluations,
                factorizations=factorizations,
                linear_solves=linear_solves,
            )

        x+=delta

        F,residual_norm=_evaluate_residual(
            residual,
            x,
        )

        residual_evaluations+=1

        if not np.isfinite(residual_norm):
            return NewtonResult(
                x=x,
                converged=False,
                iterations=iteration,
                residual_norm=np.inf,
                residual_evaluations=residual_evaluations,
                jacobian_evaluations=jacobian_evaluations,
                factorizations=factorizations,
                linear_solves=linear_solves,
            )

        if residual_norm<=tol:
            return NewtonResult(
                x=x,
                converged=True,
                iterations=iteration,
                residual_norm=residual_norm,
                residual_evaluations=residual_evaluations,
                jacobian_evaluations=jacobian_evaluations,
                factorizations=factorizations,
                linear_solves=linear_solves,
            )

    return NewtonResult(
        x=x,
        converged=False,
        iterations=max_iter,
        residual_norm=residual_norm,
        residual_evaluations=residual_evaluations,
        jacobian_evaluations=jacobian_evaluations,
        factorizations=factorizations,
        linear_solves=linear_solves,
    )


# ============================================================
# Simple Newton
# ============================================================

def simple_newton(
    residual: VectorFunction,
    frozen_jacobian_lu,
    x_0: np.ndarray,
    tol: float=1e-10,
    max_iter: int=20,
) -> NewtonResult:
    """
    Simple Newton / fixed-Jacobian iteration.

    A Jacobian factorization supplied externally is reused:

        J_* delta_j = -F(x_j),
        x_{j+1} = x_j + delta_j.

    This routine therefore performs no Jacobian evaluations and
    no factorizations itself.

    IMPORTANT:
    The construction and factorization of J_* must be counted by
    the calling time integrator when reporting total computational
    work.

    Convergence criterion:

        ||F(x_j)||_inf <= tol.
    """
    _validate_newton_parameters(
        tol,
        max_iter,
    )

    x=np.asarray(
        x_0,
        dtype=float,
    ).copy()

    if x.ndim!=1 or x.size==0:
        raise ValueError(
            "x_0 must be a nonempty one-dimensional array."
        )

    residual_evaluations=0
    linear_solves=0

    for iteration in range(max_iter+1):

        F,residual_norm=_evaluate_residual(
            residual,
            x,
        )

        residual_evaluations+=1

        # ----------------------------------------------------
        # Non-finite nonlinear iterate
        # ----------------------------------------------------

        if not np.isfinite(residual_norm):
            return NewtonResult(
                x=x,
                converged=False,
                iterations=iteration,
                residual_norm=np.inf,
                residual_evaluations=residual_evaluations,
                jacobian_evaluations=0,
                factorizations=0,
                linear_solves=linear_solves,
            )

        # ----------------------------------------------------
        # Convergence
        # ----------------------------------------------------

        if residual_norm<=tol:
            return NewtonResult(
                x=x,
                converged=True,
                iterations=iteration,
                residual_norm=residual_norm,
                residual_evaluations=residual_evaluations,
                jacobian_evaluations=0,
                factorizations=0,
                linear_solves=linear_solves,
            )

        if iteration==max_iter:
            break

        # ----------------------------------------------------
        # Fixed-Jacobian correction
        # ----------------------------------------------------

        delta=np.asarray(
            solve_factorized(
                frozen_jacobian_lu,
                -F,
            ),
            dtype=float,
        ).ravel()

        linear_solves+=1

        if delta.shape!=x.shape:
            raise ValueError(
                f"Newton correction must have shape {x.shape}, "
                f"received {delta.shape}."
            )

        if not np.all(np.isfinite(delta)):
            return NewtonResult(
                x=x,
                converged=False,
                iterations=iteration,
                residual_norm=residual_norm,
                residual_evaluations=residual_evaluations,
                jacobian_evaluations=0,
                factorizations=0,
                linear_solves=linear_solves,
            )

        x+=delta

    return NewtonResult(
        x=x,
        converged=False,
        iterations=max_iter,
        residual_norm=residual_norm,
        residual_evaluations=residual_evaluations,
        jacobian_evaluations=0,
        factorizations=0,
        linear_solves=linear_solves,
    )