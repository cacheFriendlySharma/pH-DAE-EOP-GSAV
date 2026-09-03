from pathlib import Path
import csv
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from inputs import smooth_pulse,smooth_pulse_average
from potentials import FPUBetaPotential
from system import (
    build_inverse_masses,build_topology,matrix_E,Q_matrix,B_matrix,
    J0_matrix,R0_matrix,Hamiltonian,coenergy,nonlinear_coenergy,
    g_nonlinear,dissipation,initial_state,physical_strain,
    vector_field,vector_field_jacobian,
)
from time_integrators import eop_gsav
from implicit_midpoint import implicit_midpoint
from newton_solvers import full_newton,modified_newton,simple_newton


OUT=ROOT/"benchmark"/"results"/"work_precision"
OUT.mkdir(parents=True,exist_ok=True)

# Frozen benchmark
N=2048
m_heavy,m_light=1.0,0.5
beta,beta_J,beta_R=4.0,4.0,2.0
K_M,eta_value=2.0,0.4
a_0,C_0=0.5,1.0
ell_0=ell_f=16
A_u,t_0,tau=2.0,0.5,2.0
T=5.0

dts=np.array([2e-2,1e-2,5e-3,2.5e-3,1.25e-3])
repeats=5
newton_tol=1e-11
newton_maxiter=30


# Model
potential=FPUBetaPotential(beta=beta)
inverse_masses=build_inverse_masses(N,m_light,m_heavy)
eta_vec=eta_value*np.ones(N-1)
topology=build_topology(N)

E=matrix_E(N)
Q=Q_matrix(N,inverse_masses,K_M,potential)
B=B_matrix(N,ell_f)
J0=J0_matrix(N,topology)
R0=R0_matrix(N,eta_vec,topology)
x0=initial_state(N,a_0,beta_J,ell_0)

H=lambda x: Hamiltonian(N,inverse_masses,x,potential,K_M,beta_J)
e=lambda x: coenergy(N,inverse_masses,x,potential,K_M,beta_J)
e_nl=lambda x: nonlinear_coenergy(N,x,potential,beta_J)
g=lambda x: g_nonlinear(N,inverse_masses,x,potential,beta_J,beta_R)
K=lambda x: dissipation(N,inverse_masses,x,eta_vec,beta_R)
f=lambda x: vector_field(N,inverse_masses,x,potential,K_M,eta_vec,beta_J,beta_R)
Df=lambda x: vector_field_jacobian(
    N,inverse_masses,x,potential,K_M,eta_vec,beta_J,beta_R,topology
)
u=lambda t: smooth_pulse(t,A_u,t_start=t_0,duration=tau)
u_bar=lambda a,b: smooth_pulse_average(a,b,A_u,t_start=t_0,duration=tau)

B_p=np.asarray(B[:N,0].toarray()).ravel()


# Exact index-one reduction used only for the reference trajectory
def reduced_rhs(t,y):
    p=y[:N]
    r=y[N:2*N-1]
    r_M=y[2*N-1:]

    v=inverse_masses*p
    w=v[1:]-v[:-1]
    delta=physical_strain(r,beta_J)
    gamma=np.sqrt(1.0+beta_J*r**2)

    link=potential.gradient(delta)+K_M*r_M+beta_R*w**3
    p_dot=np.zeros(N)
    p_dot[:-1]+=link
    p_dot[1:]-=link
    p_dot+=B_p*u(t)[0]

    r_dot=gamma*w
    r_M_dot=w-(K_M/eta_vec)*r_M
    return np.concatenate((p_dot,r_dot,r_M_dot))


def reference():
    y0=x0[:3*N-2]
    sol=solve_ivp(
        reduced_rhs,(0.0,T),y0,method="DOP853",
        rtol=1e-12,atol=1e-14,dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def reference_at(sol,t):
    y=sol.sol(t).T
    p=y[:,:N]
    r=y[:,N:2*N-1]
    r_M=y[:,2*N-1:]
    v=p*inverse_masses
    z=v[:,1:]-(K_M/eta_vec)*r_M
    x=np.concatenate((p,r,r_M,z),axis=1)
    H_ref=np.array([H(xi) for xi in x])
    return x,H_ref


def errors(result,ref):
    x_ref,H_ref=reference_at(ref,result.t)
    Ex=np.max(np.linalg.norm(result.x-x_ref,axis=1))
    Ex/=np.max(np.linalg.norm(x_ref,axis=1))

    EH=np.max(np.abs(result.H-H_ref))/np.max(np.abs(H_ref))
    return Ex,EH


def run_eop(dt,order):
    return eop_gsav(
        J0,R0,B,x0,0.0,T,dt,C_0,
        E=E,Q=Q,order=order,
        hamiltonian=H,nonlinear_coenergy=e_nl,
        g_function=g,dissipation_function=K,input_average=u_bar,
    )


def run_midpoint(dt,solver,frozen=False):
    return implicit_midpoint(
        E,B,x0,0.0,T,dt,solver,
        hamiltonian=H,effort_function=e,
        vector_field=f,vector_field_jacobian=Df,
        dissipation_function=K,input_function=u,
        newton_tol=newton_tol,newton_maxiter=newton_maxiter,
        use_frozen_jacobian=frozen,
    )


methods={
    "BDF1-EOP":lambda dt: run_eop(dt,1),
    "BDF2-EOP":lambda dt: run_eop(dt,2),
    "IM-full":lambda dt: run_midpoint(dt,full_newton),
    "IM-modified":lambda dt: run_midpoint(dt,modified_newton),
    "IM-simple":lambda dt: run_midpoint(dt,simple_newton,True),
}


def timed_runs(solver,dt):
    solver(dt)  # untimed warm-up

    times=[]
    result=None
    for _ in range(repeats):
        tic=time.perf_counter()
        result=solver(dt)
        times.append(time.perf_counter()-tic)

    return result,np.asarray(times)


print("Computing DOP853 reference...")
ref=reference()

records=[]
samples=[]

for dt in dts:
    print(f"\ndt={dt:.5g}")

    for name,solver in methods.items():
        print(f"  {name:11s}",end="",flush=True)
        result,times=timed_runs(solver,dt)
        Ex,EH=errors(result,ref)
        nsteps=len(result.t)-1

        if name.startswith("IM"):
            converged=bool(result.converged.all())
            mean_newton=float(result.newton_iterations.mean())
            max_newton=int(result.newton_iterations.max())
            residual_evals=int(result.residual_evaluations)
            jacobian_evals=int(result.jacobian_evaluations)
            factorizations=int(result.factorizations)
            linear_solves=int(result.linear_solves)
        else:
            converged=True
            mean_newton=max_newton=np.nan
            residual_evals=jacobian_evals=np.nan
            factorizations=1 if name=="BDF1-EOP" else 2
            linear_solves=nsteps

        record={
            "method":name,
            "dt":dt,
            "steps":nsteps,
            "state_error":Ex,
            "H_error":EH,
            "runtime_median":float(np.median(times)),
            "runtime_min":float(np.min(times)),
            "runtime_max":float(np.max(times)),
            "converged":converged,
            "mean_newton":mean_newton,
            "max_newton":max_newton,
            "residual_evaluations":residual_evals,
            "jacobian_evaluations":jacobian_evals,
            "factorizations":factorizations,
            "linear_solves":linear_solves,
        }
        records.append(record)

        for j,value in enumerate(times,1):
            samples.append({
                "method":name,"dt":dt,"repeat":j,"runtime":value
            })

        print(
            f"  Ex={Ex:.3e}  EH={EH:.3e}  "
            f"median={np.median(times):.4f}s"
        )


# Fixed-dt speedup relative to full Newton.
for row in records:
    full=next(
        r for r in records
        if r["method"]=="IM-full" and np.isclose(r["dt"],row["dt"])
    )
    row["speedup_vs_full_same_dt"]=full["runtime_median"]/row["runtime_median"]


# Main work-precision table
print("\nWork-precision table\n")
print(
    f"{'dt':>9s} {'method':>12s} {'Ex':>11s} {'EH':>11s} "
    f"{'time [s]':>10s} {'speedup*':>9s}"
)
print("-"*68)

for dt in dts:
    for r in records:
        if np.isclose(r["dt"],dt):
            print(
                f"{dt:9.3e} {r['method']:>12s} "
                f"{r['state_error']:11.3e} {r['H_error']:11.3e} "
                f"{r['runtime_median']:10.4f} "
                f"{r['speedup_vs_full_same_dt']:9.2f}"
            )

print("\n* speedup is relative to full-Newton midpoint at the SAME dt, not matched accuracy.")


# Solver-cost table
print("\nSolver-cost table\n")
print(
    f"{'dt':>9s} {'method':>12s} {'mean N':>8s} {'max N':>7s} "
    f"{'res':>8s} {'Jac':>8s} {'LU':>8s} {'solves':>9s}"
)
print("-"*79)

for dt in dts:
    for r in records:
        if not np.isclose(r["dt"],dt):
            continue

        meanN="-" if np.isnan(r["mean_newton"]) else f"{r['mean_newton']:.2f}"
        maxN="-" if np.isnan(r["max_newton"]) else f"{int(r['max_newton'])}"
        res="-" if np.isnan(r["residual_evaluations"]) else f"{int(r['residual_evaluations'])}"
        jac="-" if np.isnan(r["jacobian_evaluations"]) else f"{int(r['jacobian_evaluations'])}"

        print(
            f"{dt:9.3e} {r['method']:>12s} {meanN:>8s} {maxN:>7s} "
            f"{res:>8s} {jac:>8s} {r['factorizations']:8d} "
            f"{r['linear_solves']:9d}"
        )


# CSVs
with open(OUT/"work_precision.csv","w",newline="") as fcsv:
    writer=csv.DictWriter(fcsv,fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

with open(OUT/"timing_samples.csv","w",newline="") as fcsv:
    writer=csv.DictWriter(fcsv,fieldnames=samples[0].keys())
    writer.writeheader()
    writer.writerows(samples)

print(f"\nResults written to {OUT}")