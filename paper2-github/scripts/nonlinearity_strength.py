from pathlib import Path
import csv
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from inputs import smooth_pulse,smooth_pulse_average
from potentials import FPUBetaPotential
from system import (
    build_inverse_masses,build_topology,matrix_E,Q_matrix,B_matrix,
    J0_matrix,R0_matrix,Hamiltonian,coenergy,nonlinear_coenergy,
    g_nonlinear,dissipation,initial_state,
    vector_field,vector_field_jacobian,
)
from time_integrators import eop_gsav
from implicit_midpoint import implicit_midpoint
from newton_solvers import full_newton,modified_newton,simple_newton


OUT=ROOT/"benchmark"/"results"/"nonlinearity_strength"
OUT.mkdir(parents=True,exist_ok=True)


# ----------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------

N=2048
dt=5e-3
T=5.0
steps=int(round(T/dt))
repeats=5

lambdas=np.array([0.25,0.5,1.0,2.0])

m_heavy,m_light=1.0,0.5
K_M,eta_value=2.0,0.4
a_0,C_0=0.5,1.0
ell_0=ell_f=16
A_u,t_0,tau=2.0,0.5,2.0

newton_tol=1e-11
newton_maxiter=20


# ----------------------------------------------------------------------
# Lambda-independent objects
# ----------------------------------------------------------------------

inverse_masses=build_inverse_masses(N,m_light,m_heavy)
eta_vec=eta_value*np.ones(N-1)
topology=build_topology(N)

E=matrix_E(N)
B=B_matrix(N,ell_f)
J0=J0_matrix(N,topology)
R0=R0_matrix(N,eta_vec,topology)

u=lambda t: smooth_pulse(t,A_u,t_start=t_0,duration=tau)
u_bar=lambda a,b: smooth_pulse_average(
    a,b,A_u,t_start=t_0,duration=tau
)


def midpoint_stats(result):
    nit=np.asarray(result.newton_iterations)

    return {
        "converged":bool(np.all(result.converged)),
        "mean_newton":float(np.mean(nit)),
        "max_newton":int(np.max(nit)),
        "residual_evaluations":int(result.residual_evaluations),
        "jacobian_evaluations":int(result.jacobian_evaluations),
        "factorizations":int(result.factorizations),
        "linear_solves":int(result.linear_solves),
    }


def make_problem(lam):
    beta=4.0*lam
    beta_J=4.0*lam
    beta_R=2.0*lam

    potential=FPUBetaPotential(beta=beta)
    Q=Q_matrix(N,inverse_masses,K_M,potential)

    # Reconstruct x0 so that the physical initial strain amplitude a0
    # remains fixed as beta_J changes.
    x0=initial_state(N,a_0,beta_J,ell_0)

    H=lambda x: Hamiltonian(
        N,inverse_masses,x,potential,K_M,beta_J
    )
    e=lambda x: coenergy(
        N,inverse_masses,x,potential,K_M,beta_J
    )
    e_nl=lambda x: nonlinear_coenergy(
        N,x,potential,beta_J
    )
    g=lambda x: g_nonlinear(
        N,inverse_masses,x,potential,beta_J,beta_R
    )
    K=lambda x: dissipation(
        N,inverse_masses,x,eta_vec,beta_R
    )
    f=lambda x: vector_field(
        N,inverse_masses,x,potential,K_M,
        eta_vec,beta_J,beta_R
    )
    Df=lambda x: vector_field_jacobian(
        N,inverse_masses,x,potential,K_M,
        eta_vec,beta_J,beta_R,topology
    )

    return beta,beta_J,beta_R,Q,x0,H,e,e_nl,g,K,f,Df


def run_bdf2(Q,x0,H,e_nl,g,K):
    return eop_gsav(
        J0,R0,B,x0.copy(),0.0,T,dt,C_0,
        E=E,Q=Q,order=2,
        hamiltonian=H,
        nonlinear_coenergy=e_nl,
        g_function=g,
        dissipation_function=K,
        input_average=u_bar,
    )


def run_midpoint(x0,H,e,K,f,Df,solver,frozen=False):
    return implicit_midpoint(
        E,B,x0.copy(),0.0,T,dt,solver,
        hamiltonian=H,
        effort_function=e,
        vector_field=f,
        vector_field_jacobian=Df,
        dissipation_function=K,
        input_function=u,
        newton_tol=newton_tol,
        newton_maxiter=newton_maxiter,
        use_frozen_jacobian=frozen,
    )


# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------

rows=[]
timing_rows=[]

for lam in lambdas:
    beta,beta_J,beta_R,Q,x0,H,e,e_nl,g,K,f,Df=make_problem(lam)

    methods={
        "BDF2-EOP":lambda: run_bdf2(Q,x0,H,e_nl,g,K),
        "IM-full":lambda: run_midpoint(
            x0,H,e,K,f,Df,full_newton
        ),
        "IM-modified":lambda: run_midpoint(
            x0,H,e,K,f,Df,modified_newton
        ),
        "IM-simple":lambda: run_midpoint(
            x0,H,e,K,f,Df,simple_newton,True
        ),
    }

    print(
        f"\nlambda={lam:g}  "
        f"beta={beta:g}  beta_J={beta_J:g}  beta_R={beta_R:g}"
    )

    # One untimed warm-up per method and parameter regime.
    print("  warm-up",flush=True)
    for solver in methods.values():
        result=solver()
        if hasattr(result,"converged") and not np.all(result.converged):
            raise RuntimeError(
                f"Warm-up failed at lambda={lam:g}"
            )

    for name,solver in methods.items():
        times=[]
        representative=None

        print(f"  {name:12s}",end="",flush=True)

        for rep in range(1,repeats+1):
            tic=time.perf_counter()
            result=solver()
            elapsed=time.perf_counter()-tic

            times.append(elapsed)

            if representative is None:
                representative=result

            timing_rows.append({
                "lambda":lam,
                "beta":beta,
                "beta_J":beta_J,
                "beta_R":beta_R,
                "method":name,
                "repeat":rep,
                "runtime":elapsed,
            })

            print(f"  r{rep}:{elapsed:.4f}s",end="",flush=True)

        median=float(np.median(times))

        if name=="BDF2-EOP":
            stats={
                "converged":True,
                "mean_newton":np.nan,
                "max_newton":-1,
                "residual_evaluations":-1,
                "jacobian_evaluations":-1,
                "factorizations":2,
                "linear_solves":steps,
            }
        else:
            stats=midpoint_stats(representative)

        row={
            "lambda":lam,
            "beta":beta,
            "beta_J":beta_J,
            "beta_R":beta_R,
            "N":N,
            "unknowns":4*N-3,
            "dt":dt,
            "steps":steps,
            "method":name,
            "runtime_median":median,
            "runtime_min":float(np.min(times)),
            "runtime_max":float(np.max(times)),
            "time_per_step":median/steps,
            **stats,
        }

        rows.append(row)

        print(
            f"  median={median:.4f}s"
            +(
                ""
                if name=="BDF2-EOP"
                else
                f"  meanN={stats['mean_newton']:.2f}"
                f"  maxN={stats['max_newton']}"
            )
        )


# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------

with open(
    OUT/"nonlinearity_strength.csv","w",newline=""
) as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


with open(
    OUT/"timing_samples.csv","w",newline=""
) as f:
    writer=csv.DictWriter(f,fieldnames=timing_rows[0].keys())
    writer.writeheader()
    writer.writerows(timing_rows)


# ----------------------------------------------------------------------
# Compact terminal tables
# ----------------------------------------------------------------------

print("\nNonlinearity-strength timing\n")

print(
    f"{'lambda':>7s} {'method':>12s} "
    f"{'time [s]':>10s} {'mean N':>8s} {'max N':>7s} "
    f"{'Jac':>7s} {'LU':>7s} {'solves':>8s}"
)
print("-"*76)

for r in rows:
    meanN="-" if np.isnan(r["mean_newton"]) else f"{r['mean_newton']:.2f}"
    maxN="-" if r["max_newton"]<0 else str(r["max_newton"])
    Jac="-" if r["jacobian_evaluations"]<0 else str(r["jacobian_evaluations"])

    print(
        f"{r['lambda']:7.2f} "
        f"{r['method']:>12s} "
        f"{r['runtime_median']:10.4f} "
        f"{meanN:>8s} "
        f"{maxN:>7s} "
        f"{Jac:>7s} "
        f"{r['factorizations']:7d} "
        f"{r['linear_solves']:8d}"
    )


# Speedups relative to BDF2
print("\nRuntime ratio relative to BDF2-EOP\n")

print(
    f"{'lambda':>7s} "
    f"{'full/BDF2':>11s} "
    f"{'mod./BDF2':>11s} "
    f"{'simple/BDF2':>13s}"
)
print("-"*48)

for lam in lambdas:
    subset={
        r["method"]:r
        for r in rows
        if np.isclose(r["lambda"],lam)
    }

    tb=subset["BDF2-EOP"]["runtime_median"]

    print(
        f"{lam:7.2f} "
        f"{subset['IM-full']['runtime_median']/tb:11.2f} "
        f"{subset['IM-modified']['runtime_median']/tb:11.2f} "
        f"{subset['IM-simple']['runtime_median']/tb:13.2f}"
    )


print(f"\nResults written to {OUT}")