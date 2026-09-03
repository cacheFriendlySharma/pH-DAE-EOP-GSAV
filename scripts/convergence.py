from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from inputs import smooth_pulse,smooth_pulse_average
from plotting_config import configure_plots,figure_size
from potentials import FPUBetaPotential
from system import (
    build_inverse_masses,build_topology,matrix_E,Q_matrix,B_matrix,
    J0_matrix,R0_matrix,Hamiltonian,coenergy,nonlinear_coenergy,
    g_nonlinear,dissipation,initial_state,physical_strain,
    vector_field,vector_field_jacobian,
)
from time_integrators import eop_gsav
from implicit_midpoint import implicit_midpoint
from newton_solvers import full_newton


configure_plots()
OUT=ROOT/"benchmark"/"results"/"convergence"
OUT.mkdir(parents=True,exist_ok=True)


# Frozen benchmark
N=64
m_heavy,m_light=1.0,0.5
beta,beta_J,beta_R=4.0,4.0,2.0
K_M,eta_value=2.0,0.4
a_0,C_0=0.5,1.0
ell_0=ell_f=16
A_u,t_0,tau=2.0,0.5,2.0
T=5.0

dts=np.array([1e-2,5e-3,2.5e-3,1.25e-3])
dt_ref=dts[-1]


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
f=lambda x: vector_field(
    N,inverse_masses,x,potential,K_M,eta_vec,beta_J,beta_R
)
Df=lambda x: vector_field_jacobian(
    N,inverse_masses,x,potential,K_M,
    eta_vec,beta_J,beta_R,topology
)
u=lambda t: smooth_pulse(t,A_u,t_start=t_0,duration=tau)
u_bar=lambda a,b: smooth_pulse_average(
    a,b,A_u,t_start=t_0,duration=tau
)

B_p=np.asarray(B[:N,0].toarray()).ravel()


# Reduced index-one reference system
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


def reduced_reference(t_eval,rtol,atol):
    y0=x0[:3*N-2]

    sol=solve_ivp(
        reduced_rhs,(0.0,T),y0,
        method="DOP853",t_eval=t_eval,
        rtol=rtol,atol=atol,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    p=sol.y[:N].T
    r=sol.y[N:2*N-1].T
    r_M=sol.y[2*N-1:].T

    v=p*inverse_masses
    z=v[:,1:]-(K_M/eta_vec)*r_M

    x=np.concatenate((p,r,r_M,z),axis=1)
    H_ref=np.array([H(xi) for xi in x])

    return x,H_ref


def run_midpoint(dt):
    return implicit_midpoint(
        E,B,x0,0.0,T,dt,full_newton,
        hamiltonian=H,effort_function=e,
        vector_field=f,vector_field_jacobian=Df,
        dissipation_function=K,input_function=u,
        newton_tol=1e-11,newton_maxiter=20,
    )


def run_eop(dt,order):
    return eop_gsav(
        J0,R0,B,x0,0.0,T,dt,C_0,
        E=E,Q=Q,order=order,
        hamiltonian=H,
        nonlinear_coenergy=e_nl,
        g_function=g,
        dissipation_function=K,
        input_average=u_bar,
    )


def errors(result,x_ref,H_ref):
    stride=int(round(result.t[1]/dt_ref))
    idx=np.arange(0,len(x_ref),stride)

    xr=x_ref[idx]
    Hr=H_ref[idx]

    dx=np.linalg.norm(result.x-xr,axis=1)
    Ex=np.max(dx)/np.max(np.linalg.norm(xr,axis=1))
    EH=np.max(np.abs(result.H-Hr))/np.max(np.abs(Hr))

    return Ex,EH


# Independent reference + tolerance check
t_ref=np.arange(0.0,T+0.5*dt_ref,dt_ref)

print("\nComputing DOP853 reference...")
x_ref,H_ref=reduced_reference(t_ref,rtol=1e-12,atol=1e-14)

print("Computing tightened DOP853 reference...")
x_check,H_check=reduced_reference(t_ref,rtol=2e-13,atol=2e-15)

ref_x=np.max(np.linalg.norm(x_ref-x_check,axis=1))/np.max(
    np.linalg.norm(x_check,axis=1)
)
ref_H=np.max(np.abs(H_ref-H_check))/np.max(np.abs(H_check))

print(f"Reference check: Ex={ref_x:.3e}, EH={ref_H:.3e}\n")


methods={
    "BDF1":lambda dt: run_eop(dt,1),
    "BDF2":lambda dt: run_eop(dt,2),
    "Midpoint":run_midpoint,
}

data={}

for name,solver in methods.items():
    Ex=np.zeros(len(dts))
    EH=np.zeros(len(dts))

    for j,dt in enumerate(dts):
        print(f"{name:8s} dt={dt:.5g}")
        result=solver(dt)
        Ex[j],EH[j]=errors(result,x_ref,H_ref)

    rates_x=np.log(Ex[:-1]/Ex[1:])/np.log(dts[:-1]/dts[1:])
    rates_H=np.log(EH[:-1]/EH[1:])/np.log(dts[:-1]/dts[1:])
    data[name]=(Ex,EH,rates_x,rates_H)


# Reference accuracy relative to finest second-order error
min_x=min(data["BDF2"][0][-1],data["Midpoint"][0][-1])
min_H=min(data["BDF2"][1][-1],data["Midpoint"][1][-1])

if ref_x>0.02*min_x or ref_H>0.02*min_H:
    raise RuntimeError(
        f"Reference not sufficiently resolved: "
        f"Ex={ref_x:.3e}, EH={ref_H:.3e}."
    )


# Tables
print("\nTemporal convergence\n")
print(
    f"{'dt':>10s} "
    f"{'BDF1 Ex':>12s} {'BDF1 EH':>12s} "
    f"{'BDF2 Ex':>12s} {'BDF2 EH':>12s} "
    f"{'IM Ex':>12s} {'IM EH':>12s}"
)
print("-"*88)

for j,dt in enumerate(dts):
    print(
        f"{dt:10.3e} "
        f"{data['BDF1'][0][j]:12.3e} {data['BDF1'][1][j]:12.3e} "
        f"{data['BDF2'][0][j]:12.3e} {data['BDF2'][1][j]:12.3e} "
        f"{data['Midpoint'][0][j]:12.3e} "
        f"{data['Midpoint'][1][j]:12.3e}"
    )

print("\nSuccessive observed orders")

for name,(Ex,EH,px,pH) in data.items():
    print(
        f"{name:8s}  state: "
        +", ".join(f"{p:.3f}" for p in px)
        +"   Hamiltonian: "
        +", ".join(f"{p:.3f}" for p in pH)
    )


# Save data
rows=np.column_stack((
    dts,
    data["BDF1"][0],data["BDF1"][1],
    data["BDF2"][0],data["BDF2"][1],
    data["Midpoint"][0],data["Midpoint"][1],
))

header=(
    "dt,BDF1_Ex,BDF1_EH,"
    "BDF2_Ex,BDF2_EH,"
    "Midpoint_Ex,Midpoint_EH"
)

np.savetxt(
    OUT/"convergence.csv",
    rows,
    delimiter=",",
    header=header,
    comments="",
)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

plot_idx=np.argsort(dts)
h=dts[plot_idx]

xticklabels=[
    r"$1.25\times10^{-3}$",
    r"$2.5\times10^{-3}$",
    r"$5\times10^{-3}$",
    r"$10^{-2}$",
]

styles={
    "BDF1":("o-","BDF-1-EOP"),
    "BDF2":("s-","BDF-2-EOP"),
    "Midpoint":("^-","IMP"),
}


def convergence_plot(error_index,ylabel,filename):
    fig,ax=plt.subplots(figsize=figure_size("single"))

    method_lines=[]
    method_labels=[]

    for name in methods:
        values=data[name][error_index][plot_idx]
        style,label=styles[name]
        line,=ax.loglog(h,values,style,label=label)
        method_lines.append(line)
        method_labels.append(label)

    e1=data["BDF1"][error_index][plot_idx]
    e2=data["BDF2"][error_index][plot_idx]

    ref1=0.62*e1[-1]*(h/h[-1])
    ref2=0.58*e2[-1]*(h/h[-1])**2

    order1,=ax.loglog(
        h,ref1,"--",linewidth=1.0,
        label=r"$\mathcal{O}(\delta t)$",
    )
    order2,=ax.loglog(
        h,ref2,"--",linewidth=1.0,
        label=r"$\mathcal{O}(\delta t^2)$",
    )

    ax.set_xlabel(r"$\delta t$")
    ax.set_ylabel(ylabel)

    ax.set_xticks(h)
    ax.set_xticklabels([
        r"$1.25\times10^{-3}$",
        r"$2.5\times10^{-3}$",
        r"$5\times10^{-3}$",
        r"$10^{-2}$",
    ])
    ax.minorticks_off()
    ax.grid(True,which="major",alpha=.3)

    # Order-reference legend inside the axes
    ax.legend(
        [order1,order2],
        [r"$\mathcal{O}(\delta t)$",
         r"$\mathcal{O}(\delta t^2)$"],
        loc="lower right",
        frameon=False,
        fontsize=8,
        handlelength=1.7,
    )

    # Solver/color legend below the axes
    fig.legend(
        method_lines,
        method_labels,
        loc="lower center",
        bbox_to_anchor=(0.5,-0.075),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=1.8,
    )

    fig.subplots_adjust(bottom=0.31)

    # Reserve room for the external solver legend
    fig.subplots_adjust(bottom=0.25)

    fig.savefig(
        OUT/filename,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)

convergence_plot(
    0,
    r"$\varepsilon_x$",
    "state_convergence.pdf",
)

convergence_plot(
    1,
    r"$\varepsilon_H$",
    "hamiltonian_convergence.pdf",
)

print(f"\nResults written to {OUT}")