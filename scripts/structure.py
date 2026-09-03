from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from inputs import smooth_pulse,smooth_pulse_average
from plotting_config import configure_plots,figure_size
from potentials import FPUBetaPotential
from system import (
    build_inverse_masses,build_topology,matrix_E,Q_matrix,B_matrix,
    J0_matrix,R0_matrix,Hamiltonian,coenergy,nonlinear_coenergy,
    g_nonlinear,dissipation,initial_state,vector_field,vector_field_jacobian,
)
from time_integrators import eop_gsav
from implicit_midpoint import implicit_midpoint
from newton_solvers import full_newton


configure_plots()
OUT=ROOT/"benchmark"/"results"/"structure"
OUT.mkdir(parents=True,exist_ok=True)


# Frozen benchmark
N=64
m_heavy,m_light=1.0,0.5
beta,beta_J,beta_R_base=4.0,4.0,2.0
K_M,eta_base=2.0,0.4
a_0,C_0=0.5,1.0
ell_0=ell_f=16
A_u,t_0,tau=2.0,0.5,2.0
T=5.0

dts=np.array([1e-2,5e-3,2.5e-3])
eps_R=np.array([1.0,0.25,0.1,0.025,0.01])
stress_dt=1e-2


# Fixed components
potential=FPUBetaPotential(beta=beta)
inverse_masses=build_inverse_masses(N,m_light,m_heavy)
topology=build_topology(N)

E=matrix_E(N)
Q=Q_matrix(N,inverse_masses,K_M,potential)
B=B_matrix(N,ell_f)
J0=J0_matrix(N,topology)
x0=initial_state(N,a_0,beta_J,ell_0)

H=lambda x: Hamiltonian(N,inverse_masses,x,potential,K_M,beta_J)
e=lambda x: coenergy(N,inverse_masses,x,potential,K_M,beta_J)
e_nl=lambda x: nonlinear_coenergy(N,x,potential,beta_J)
u=lambda t: smooth_pulse(t,A_u,t_start=t_0,duration=tau)
u_bar=lambda a,b: smooth_pulse_average(a,b,A_u,t_start=t_0,duration=tau)


def operators(eps=1.0):
    eta=(eta_base/eps)*np.ones(N-1)
    beta_R=eps*beta_R_base
    R0=R0_matrix(N,eta,topology)

    g=lambda x: g_nonlinear(N,inverse_masses,x,potential,beta_J,beta_R)
    K=lambda x: dissipation(N,inverse_masses,x,eta,beta_R)
    f=lambda x: vector_field(N,inverse_masses,x,potential,K_M,eta,beta_J,beta_R)
    Df=lambda x: vector_field_jacobian(
        N,inverse_masses,x,potential,K_M,eta,beta_J,beta_R,topology
    )
    return R0,g,K,f,Df


def run_eop(dt,order,eps=1.0):
    R0,g,K,_,_=operators(eps)
    return eop_gsav(
        J0,R0,B,x0,0.0,T,dt,C_0,
        E=E,Q=Q,order=order,
        hamiltonian=H,nonlinear_coenergy=e_nl,
        g_function=g,dissipation_function=K,input_average=u_bar,
    )


def run_midpoint(dt,eps=1.0):
    _,_,K,f,Df=operators(eps)
    return implicit_midpoint(
        E,B,x0,0.0,T,dt,full_newton,
        hamiltonian=H,effort_function=e,
        vector_field=f,vector_field_jacobian=Df,
        dissipation_function=K,input_function=u,
        newton_tol=1e-11,newton_maxiter=20,
    )


def diagnostics(result,dt,r=None):
    dH=np.diff(result.H)
    work_step=dt*result.power
    diss_step=dt*result.dissipation
    balance_step=dH-work_step+diss_step

    W=np.concatenate(([0.0],np.cumsum(work_step)))
    D=np.concatenate(([0.0],np.cumsum(diss_step)))

    physical_step=np.maximum(dH-work_step,0.0)
    physical_cum=np.maximum(result.H-result.H[0]-W,0.0)
    balance_cum=result.H-result.H[0]-W+D

    native=np.nan
    if r is not None:
        native=float(np.max(np.maximum(np.diff(r)-work_step,0.0)))

    return {
        "native":native,
        "physical_step":float(np.max(physical_step)),
        "physical_cum":float(np.max(physical_cum)),
        "balance_max":float(np.max(balance_step)),
        "balance_min":float(np.min(balance_step)),
        "balance_abs":float(np.max(np.abs(balance_step))),
        "balance_cum_abs":float(np.max(np.abs(balance_cum))),
        "balance_history":balance_cum,
        "total_dissipation":float(np.sum(diss_step)),
    }


# ----------------------------------------------------------------------
# Frozen dissipative benchmark
# ----------------------------------------------------------------------

methods={
    "BDF1":lambda dt: run_eop(dt,1),
    "BDF2":lambda dt: run_eop(dt,2),
    "Midpoint":run_midpoint,
}

data={name:[] for name in methods}
history={}

for dt in dts:
    for name,solver in methods.items():
        print(f"{name:8s} dt={dt:.5g}")
        result=solver(dt)
        diag=diagnostics(result,dt,result.r if name.startswith("BDF") else None)
        data[name].append(diag)

        if dt==dts[0]:
            history[name]=(result.t,diag["balance_history"])


print("\nStructure diagnostics: frozen dissipative benchmark\n")
print(
    f"{'method':9s} {'dt':>9s} {'native':>11s} {'phys.cum':>11s} "
    f"{'bal.max':>11s} {'bal.min':>11s} {'|bal|':>11s} {'cum.|bal|':>11s}"
)
print("-"*93)

with open(OUT/"structure.csv","w") as fcsv:
    fcsv.write(
        "method,dt,native_passivity,physical_step_passivity,"
        "physical_cumulative_passivity,balance_step_max,balance_step_min,"
        "balance_step_abs,balance_cumulative_abs,total_dissipation\n"
    )

    for name in methods:
        for dt,diag in zip(dts,data[name]):
            print(
                f"{name:9s} {dt:9.3e} {diag['native']:11.3e} "
                f"{diag['physical_cum']:11.3e} "
                f"{diag['balance_max']:11.3e} {diag['balance_min']:11.3e} "
                f"{diag['balance_abs']:11.3e} {diag['balance_cum_abs']:11.3e}"
            )

            fcsv.write(
                f"{name},{dt:.16e},{diag['native']:.16e},"
                f"{diag['physical_step']:.16e},{diag['physical_cum']:.16e},"
                f"{diag['balance_max']:.16e},{diag['balance_min']:.16e},"
                f"{diag['balance_abs']:.16e},{diag['balance_cum_abs']:.16e},"
                f"{diag['total_dissipation']:.16e}\n"
            )


fig,ax=plt.subplots(figsize=figure_size("single"))
for name in methods:
    ax.loglog(dts,[d["balance_cum_abs"] for d in data[name]],"o-",label=name)

ax.set_xlabel(r"$\Delta t$")
ax.set_ylabel(r"$\max_n |D_H^n|$")
ax.legend()
fig.savefig(OUT/"balance_defect.pdf")
plt.close(fig)


fig,ax=plt.subplots(figsize=figure_size("single"))
for name,(t,balance) in history.items():
    ax.plot(t,balance,label=name)

ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$D_H(t)$")
ax.legend()
fig.savefig(OUT/"balance_history.pdf")
plt.close(fig)


# ----------------------------------------------------------------------
# Near-conservative homotopy
# beta_R -> eps beta_R, eta -> eta/eps
# ----------------------------------------------------------------------

stress=[]

print("\nNear-conservative structure stress test\n")
print(
    f"{'eps':>8s} {'method':>9s} {'native':>11s} {'phys.step':>11s} "
    f"{'phys.cum':>11s} {'bal.max':>11s} {'bal.min':>11s} {'D/H0':>10s}"
)
print("-"*93)

for eps in eps_R:
    for name in ("BDF2","Midpoint"):
        if name=="BDF2":
            result=run_eop(stress_dt,2,eps)
            diag=diagnostics(result,stress_dt,result.r)
        else:
            result=run_midpoint(stress_dt,eps)
            diag=diagnostics(result,stress_dt)

        D_ratio=diag["total_dissipation"]/result.H[0]
        stress.append((eps,name,diag,D_ratio))

        print(
            f"{eps:8.3f} {name:>9s} {diag['native']:11.3e} "
            f"{diag['physical_step']:11.3e} {diag['physical_cum']:11.3e} "
            f"{diag['balance_max']:11.3e} {diag['balance_min']:11.3e} "
            f"{D_ratio:10.3e}"
        )


with open(OUT/"near_conservative.csv","w") as fcsv:
    fcsv.write(
        "epsilon,method,native_passivity,physical_step_passivity,"
        "physical_cumulative_passivity,balance_step_max,balance_step_min,"
        "total_dissipation_over_H0\n"
    )

    for eps,name,diag,D_ratio in stress:
        fcsv.write(
            f"{eps:.16e},{name},{diag['native']:.16e},"
            f"{diag['physical_step']:.16e},{diag['physical_cum']:.16e},"
            f"{diag['balance_max']:.16e},{diag['balance_min']:.16e},"
            f"{D_ratio:.16e}\n"
        )


# Physical passivity
fig,ax=plt.subplots(figsize=figure_size("single"))

for name in ("BDF2","Midpoint"):
    rows=[row for row in stress if row[1]==name]
    ax.semilogx(
        [row[0] for row in rows],
        [row[2]["physical_step"] for row in rows],
        "o-",label=name,
    )

ax.set_xlabel(r"$\varepsilon$")
ax.set_ylabel("maximum physical passivity violation")
ax.invert_xaxis()
ax.legend()
fig.savefig(OUT/"near_conservative_passivity.pdf")
plt.close(fig)


# Signed local balance defect
fig,ax=plt.subplots(figsize=figure_size("single"))

for name in ("BDF2","Midpoint"):
    rows=[row for row in stress if row[1]==name]
    eps=[row[0] for row in rows]

    ax.semilogx(
        eps,[row[2]["balance_max"] for row in rows],
        "o-",label=f"{name}: maximum",
    )
    ax.semilogx(
        eps,[row[2]["balance_min"] for row in rows],
        "o--",label=f"{name}: minimum",
    )

ax.axhline(0.0,linewidth=0.8)
ax.set_xlabel(r"$\varepsilon$")
ax.set_ylabel("local physical balance defect")
ax.invert_xaxis()
ax.legend()
fig.savefig(OUT/"near_conservative_balance.pdf")
plt.close(fig)


# Dissipation along homotopy
fig,ax=plt.subplots(figsize=figure_size("single"))

for name in ("BDF2","Midpoint"):
    rows=[row for row in stress if row[1]==name]
    ax.loglog(
        [row[0] for row in rows],
        [row[3] for row in rows],
        "o-",label=name,
    )

ax.set_xlabel(r"$\varepsilon$")
ax.set_ylabel(r"$\mathcal{D}(T)/H(0)$")
ax.invert_xaxis()
ax.legend()
fig.savefig(OUT/"near_conservative_dissipation.pdf")
plt.close(fig)

print(f"\nResults written to {OUT}")