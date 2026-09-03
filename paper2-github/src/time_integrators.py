from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps

from linear_solvers import factorize,solve_factorized
from system import bdf1_system_matrix,bdf2_system_matrix,full_coenergy,linear_A


ScalarFunction=Callable[[np.ndarray],float]
VectorFunction=Callable[[np.ndarray],np.ndarray]
InputAverage=Callable[[float,float],np.ndarray]


@dataclass
class BDF1StepResult:
    x_bar: np.ndarray
    x: np.ndarray
    H_bar: float
    H: float
    r_tilde: float
    r: float
    xi: float
    eta: float
    power: float
    dissipation: float
    budget: float
    primal_residual: float
    projection_active: bool


@dataclass
class BDFkEOPGSAVResult:
    t: np.ndarray
    x: np.ndarray
    H: np.ndarray
    H_bar: np.ndarray
    r: np.ndarray
    r_tilde: np.ndarray
    xi: np.ndarray
    eta: np.ndarray
    power: np.ndarray
    dissipation: np.ndarray
    budget: np.ndarray
    primal_residual: np.ndarray
    projection_active: np.ndarray


def _validate(E,J0,R0,B,Q,x_0,t_s,t_e,dt,C_0,min_steps):
    if dt<=0.0 or t_e<=t_s or C_0<=0.0:
        raise ValueError("Require dt>0, t_e>t_s and C_0>0.")

    x_0=np.asarray(x_0,dtype=float).ravel()
    n=x_0.size

    if not n:
        raise ValueError("x_0 must be nonempty.")

    for name,M in (("E",E),("J0",J0),("R0",R0),("Q",Q)):
        if M.shape!=(n,n):
            raise ValueError(f"{name} has incompatible shape {M.shape}.")

    if B.ndim!=2 or B.shape[0]!=n:
        raise ValueError(f"B has incompatible shape {B.shape}.")

    steps_float=(t_e-t_s)/dt
    steps=int(round(steps_float))

    if steps<min_steps or not np.isclose(steps_float,steps,rtol=1e-12,atol=1e-14):
        raise ValueError("Time interval must contain an integer number of steps.")

    return steps,t_s+dt*np.arange(steps+1),x_0


def _allocate(steps,n):
    return {
        "x":np.zeros((steps+1,n)),
        "H":np.zeros(steps+1),
        "H_bar":np.zeros(steps),
        "r":np.zeros(steps+1),
        "r_tilde":np.zeros(steps+1),
        "xi":np.ones(steps+1),
        "eta":np.ones(steps+1),
        "power":np.zeros(steps),
        "dissipation":np.zeros(steps),
        "budget":np.zeros(steps+1),
        "primal_residual":np.zeros(steps),
        "projection_active":np.zeros(steps,dtype=bool),
    }


def _input_average(input_average,t0,t1,m):
    if input_average is None:
        return np.zeros(m)

    u=np.asarray(input_average(t0,t1),dtype=float).ravel()
    if u.shape!=(m,):
        raise ValueError(f"Input average must have shape {(m,)}, received {u.shape}.")
    return u


def _bdf1_step(E,B,Q,A,dt,C_0,M_factor,x_n,r_n,u_bar,
               hamiltonian,nonlinear_coenergy,g_function,dissipation_function):
    g_n=np.asarray(g_function(x_n),dtype=float).ravel()
    rhs=E@x_n+dt*(B@u_bar-g_n)
    x_bar=solve_factorized(M_factor,rhs)

    e_bar=full_coenergy(Q,x_bar,nonlinear_coenergy)
    y_bar=np.asarray(B.T@e_bar).ravel()
    H_bar=float(hamiltonian(x_bar))
    H_shift=H_bar+C_0

    if H_shift<=0.0:
        raise FloatingPointError(f"H_bar+C_0={H_shift:.6e} must be positive.")

    K=float(dissipation_function(x_bar))
    P=float(u_bar@y_bar)

    r_tilde=(r_n+dt*P)/(1.0+dt*K/H_shift)
    xi=r_tilde/H_shift
    eta=1.0-(1.0-xi)**2
    x=eta*x_bar

    H=float(hamiltonian(x))
    budget=r_n+dt*P
    H_projected=H+C_0
    active=budget<H_projected
    r=min(H_projected,budget)

    primal=E@(x_bar-x_n)/dt+A@x_bar+g_n-B@u_bar

    return BDF1StepResult(
        x_bar=x_bar,x=x,H_bar=H_bar,H=H,r_tilde=r_tilde,r=r,
        xi=xi,eta=eta,power=P,dissipation=K,budget=budget,
        primal_residual=np.linalg.norm(primal,ord=np.inf),
        projection_active=active,
    )


def eop_gsav(
    J0,R0,B,x_0,t_s,t_e,dt,C_0,*,
    hamiltonian: ScalarFunction,nonlinear_coenergy: VectorFunction,
    g_function: VectorFunction,dissipation_function: ScalarFunction,
    E=None,Q=None,order=2,input_average: InputAverage|None=None,
):
    """
    BDF1/BDF2 EOP-GSAV for

        E x' + A x + g(x) = B u,   A = -(J0-R0)Q,

    with e(x)=Qx+e_nl(x) and full physical dissipation K(x).
    """
    if order not in (1,2):
        raise ValueError("order must be 1 or 2.")

    x_0=np.asarray(x_0,dtype=float).ravel()
    n=x_0.size

    if E is None:
        E=sps.eye(n,format="csr")
    if Q is None:
        Q=sps.csr_matrix((n,n))

    steps,t,x_0=_validate(
        E,J0,R0,B,Q,x_0,t_s,t_e,dt,C_0,1 if order==1 else 2
    )

    arrays=_allocate(steps,n)
    A=linear_A(J0,R0,Q)
    M1=factorize(bdf1_system_matrix(E,A,dt))
    M2=factorize(bdf2_system_matrix(E,A,dt)) if order==2 else None

    x,H,H_bar=arrays["x"],arrays["H"],arrays["H_bar"]
    r,r_tilde=arrays["r"],arrays["r_tilde"]
    xi,eta=arrays["xi"],arrays["eta"]
    power,K=arrays["power"],arrays["dissipation"]
    budget=arrays["budget"]
    residual,active=arrays["primal_residual"],arrays["projection_active"]

    x[0]=x_0
    H[0]=float(hamiltonian(x_0))
    r[0]=r_tilde[0]=budget[0]=H[0]+C_0

    if order==1:
        for nstep in range(steps):
            u_bar=_input_average(input_average,t[nstep],t[nstep+1],B.shape[1])

            step=_bdf1_step(
                E,B,Q,A,dt,C_0,M1,x[nstep],r[nstep],u_bar,
                hamiltonian,nonlinear_coenergy,g_function,dissipation_function,
            )

            x[nstep+1],H[nstep+1],H_bar[nstep]=step.x,step.H,step.H_bar
            r[nstep+1],r_tilde[nstep+1]=step.r,step.r_tilde
            xi[nstep+1],eta[nstep+1]=step.xi,step.eta
            power[nstep],K[nstep]=step.power,step.dissipation
            budget[nstep+1]=step.budget
            residual[nstep],active[nstep]=step.primal_residual,step.projection_active

        return BDFkEOPGSAVResult(t=t,**arrays)

    y_bar=np.zeros((steps,B.shape[1]))
    u_prev=_input_average(input_average,t[0],t[1],B.shape[1])

    startup=_bdf1_step(
        E,B,Q,A,dt,C_0,M1,x[0],r[0],u_prev,
        hamiltonian,nonlinear_coenergy,g_function,dissipation_function,
    )

    x[1],H[1],H_bar[0]=startup.x,startup.H,startup.H_bar
    r[1],r_tilde[1]=startup.r,startup.r_tilde
    xi[1],eta[1]=startup.xi,startup.eta
    power[0],K[0]=startup.power,startup.dissipation
    budget[1]=startup.budget
    residual[0],active[0]=startup.primal_residual,startup.projection_active

    e_bar=full_coenergy(Q,startup.x_bar,nonlinear_coenergy)
    y_bar[0]=np.asarray(B.T@e_bar).ravel()

    for nstep in range(1,steps):
        u_bar=_input_average(input_average,t[nstep],t[nstep+1],B.shape[1])
        u_bdf2=1.5*u_bar-0.5*u_prev

        x_ext=2.0*x[nstep]-x[nstep-1]
        g_ext=np.asarray(g_function(x_ext),dtype=float).ravel()

        rhs=(
            2.0*(E@x[nstep])-0.5*(E@x[nstep-1])
            +dt*(B@u_bdf2-g_ext)
        )
        x_bar=solve_factorized(M2,rhs)

        e_bar=full_coenergy(Q,x_bar,nonlinear_coenergy)
        y_bar[nstep]=np.asarray(B.T@e_bar).ravel()

        H_bar[nstep]=float(hamiltonian(x_bar))
        H_shift=H_bar[nstep]+C_0

        if H_shift<=0.0:
            raise FloatingPointError(
                f"H_bar+C_0={H_shift:.6e} at step {nstep} must be positive."
            )

        K[nstep]=float(dissipation_function(x_bar))
        power[nstep]=float(u_bar@(0.5*(y_bar[nstep]+y_bar[nstep-1])))

        r_tilde[nstep+1]=(r[nstep]+dt*power[nstep])/(1.0+dt*K[nstep]/H_shift)
        xi[nstep+1]=r_tilde[nstep+1]/H_shift
        eta[nstep+1]=1.0-(1.0-xi[nstep+1])**3

        x[nstep+1]=eta[nstep+1]*x_bar
        H[nstep+1]=float(hamiltonian(x[nstep+1]))

        budget[nstep+1]=r[nstep]+dt*power[nstep]
        H_projected=H[nstep+1]+C_0
        active[nstep]=budget[nstep+1]<H_projected
        r[nstep+1]=min(H_projected,budget[nstep+1])

        primal=(
            (1.5*(E@x_bar)-2.0*(E@x[nstep])+0.5*(E@x[nstep-1]))/dt
            +A@x_bar+g_ext-B@u_bdf2
        )
        residual[nstep]=np.linalg.norm(primal,ord=np.inf)
        u_prev=u_bar

    return BDFkEOPGSAVResult(t=t,**arrays)