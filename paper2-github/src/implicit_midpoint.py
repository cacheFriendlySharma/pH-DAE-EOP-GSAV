from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from linear_solvers import factorize
from newton_solvers import NewtonResult


VectorFunction=Callable[[np.ndarray],np.ndarray]
ScalarFunction=Callable[[np.ndarray],float]
InputFunction=Callable[[float],np.ndarray]


@dataclass
class ImplicitMidpointResult:
    t: np.ndarray
    x: np.ndarray
    H: np.ndarray
    power: np.ndarray
    dissipation: np.ndarray
    residual_norm: np.ndarray
    newton_iterations: np.ndarray
    converged: np.ndarray
    residual_evaluations: int
    jacobian_evaluations: int
    factorizations: int
    linear_solves: int


# ============================================================
# Midpoint nonlinear system
# ============================================================

def midpoint_residual(z,x_n,u_mid,E,B,vector_field,dt):
    x_mid=0.5*(z+x_n)

    return np.asarray(
        E@(z-x_n)/dt-vector_field(x_mid)-B@u_mid,
        dtype=float,
    ).ravel()


def midpoint_jacobian(z,x_n,E,vector_field_jacobian,dt):
    x_mid=0.5*(z+x_n)

    return E/dt-0.5*vector_field_jacobian(x_mid)


# ============================================================
# Implicit midpoint integrator
# ============================================================

def implicit_midpoint(
    E,B,x_0,t_s,t_e,dt,newton_solver,*,
    hamiltonian: ScalarFunction,effort_function: VectorFunction,
    vector_field: VectorFunction,vector_field_jacobian: VectorFunction,
    dissipation_function: ScalarFunction,input_function: InputFunction|None=None,
    newton_tol=1e-10,newton_maxiter=20,use_frozen_jacobian=False,
):
    """
    Implicit midpoint for

        E x' = f(x)+Bu,

    with

        f(x)=(J(x)-R(x))e(x).

    Full and modified Newton use the exact midpoint Jacobian

        E/dt - 1/2 Df(x_mid).

    With use_frozen_jacobian=True, a single Jacobian factorization
    is constructed at the initial state and reused for the complete
    simulation.
    """
    if dt<=0.0:
        raise ValueError("dt must be positive.")
    if t_e<=t_s:
        raise ValueError("t_e must be greater than t_s.")

    x_0=np.asarray(x_0,dtype=float).ravel()
    if x_0.size==0:
        raise ValueError("x_0 must be nonempty.")

    state_size=x_0.size

    if E.shape!=(state_size,state_size):
        raise ValueError(f"E must have shape {(state_size,state_size)}.")
    if B.shape[0]!=state_size:
        raise ValueError("B has incompatible state dimension.")

    steps_float=(t_e-t_s)/dt
    num_steps=int(round(steps_float))

    if not np.isclose(steps_float,num_steps,rtol=1e-12,atol=1e-14):
        raise ValueError("The time interval must be an integer multiple of dt.")

    t=t_s+dt*np.arange(num_steps+1)

    if input_function is None:
        input_function=lambda time: np.zeros(B.shape[1])

    x=np.zeros((num_steps+1,state_size))
    H=np.zeros(num_steps+1)
    power=np.zeros(num_steps)
    dissipation=np.zeros(num_steps)
    residual_norm=np.zeros(num_steps)
    newton_iterations=np.zeros(num_steps,dtype=int)
    converged=np.zeros(num_steps,dtype=bool)

    total_residual_evaluations=0
    total_jacobian_evaluations=0
    total_factorizations=0
    total_linear_solves=0

    x[0]=x_0
    H[0]=float(hamiltonian(x[0]))

    # --------------------------------------------------------
    # Simple Newton: one globally frozen Jacobian
    # --------------------------------------------------------

    frozen_jacobian_lu=None

    if use_frozen_jacobian:
        J_frozen=midpoint_jacobian(
            x[0],x[0],E,vector_field_jacobian,dt
        )
        frozen_jacobian_lu=factorize(J_frozen)

        total_jacobian_evaluations+=1
        total_factorizations+=1

    # --------------------------------------------------------
    # Time stepping
    # --------------------------------------------------------

    for n in range(num_steps):
        x_n=x[n]
        t_mid=0.5*(t[n]+t[n+1])

        u_mid=np.asarray(input_function(t_mid),dtype=float).ravel()
        if u_mid.shape!=(B.shape[1],):
            raise ValueError(
                f"Input must have shape {(B.shape[1],)}, received {u_mid.shape}."
            )

        def residual(z):
            return midpoint_residual(
                z,x_n,u_mid,E,B,vector_field,dt
            )

        def jacobian(z):
            return midpoint_jacobian(
                z,x_n,E,vector_field_jacobian,dt
            )

        if use_frozen_jacobian:
            result=newton_solver(
                residual=residual,frozen_jacobian_lu=frozen_jacobian_lu,
                x_0=x_n,tol=newton_tol,max_iter=newton_maxiter,
            )
        else:
            result=newton_solver(
                residual=residual,jacobian=jacobian,x_0=x_n,
                tol=newton_tol,max_iter=newton_maxiter,
            )

        if not result.converged:
            raise RuntimeError(
                f"{newton_solver.__name__} failed at step {n}, "
                f"t={t[n+1]:.6e}, residual={result.residual_norm:.3e}."
            )

        x[n+1]=result.x
        H[n+1]=float(hamiltonian(x[n+1]))

        x_mid=0.5*(x[n+1]+x[n])
        e_mid=np.asarray(effort_function(x_mid),dtype=float).ravel()
        y_mid=np.asarray(B.T@e_mid,dtype=float).ravel()

        power[n]=float(u_mid@y_mid)
        dissipation[n]=float(dissipation_function(x_mid))

        if dissipation[n]<-1e-12:
            raise FloatingPointError(
                f"Negative dissipation at step {n}: {dissipation[n]:.6e}."
            )

        residual_norm[n]=result.residual_norm
        newton_iterations[n]=result.iterations
        converged[n]=result.converged

        total_residual_evaluations+=result.residual_evaluations
        total_jacobian_evaluations+=result.jacobian_evaluations
        total_factorizations+=result.factorizations
        total_linear_solves+=result.linear_solves

    return ImplicitMidpointResult(
        t=t,x=x,H=H,power=power,dissipation=dissipation,
        residual_norm=residual_norm,newton_iterations=newton_iterations,
        converged=converged,
        residual_evaluations=total_residual_evaluations,
        jacobian_evaluations=total_jacobian_evaluations,
        factorizations=total_factorizations,
        linear_solves=total_linear_solves,
    )