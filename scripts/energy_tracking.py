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
    g_nonlinear,dissipation,initial_state,
    vector_field,vector_field_jacobian,
)
from time_integrators import eop_gsav
from implicit_midpoint import implicit_midpoint
from newton_solvers import full_newton


configure_plots()
OUT=ROOT/"benchmark"/"results"/"energy_tracking"
OUT.mkdir(parents=True,exist_ok=True)


# Frozen benchmark
N,dt,T=64,1e-2,5.0
m_heavy,m_light=1.0,0.5
beta,beta_J,beta_R=4.0,4.0,2.0
K_M,eta_value=2.0,0.4
a_0,C_0=0.5,1.0
ell_0=ell_f=16
A_u,t_0,tau=2.0,0.5,2.0


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


def run_eop(order):
    return eop_gsav(
        J0,R0,B,x0,0.0,T,dt,C_0,
        E=E,Q=Q,order=order,
        hamiltonian=H,
        nonlinear_coenergy=e_nl,
        g_function=g,
        dissipation_function=K,
        input_average=u_bar,
    )


def run_midpoint():
    return implicit_midpoint(
        E,B,x0,0.0,T,dt,full_newton,
        hamiltonian=H,
        effort_function=e,
        vector_field=f,
        vector_field_jacobian=Df,
        dissipation_function=K,
        input_function=u,
        newton_tol=1e-11,
        newton_maxiter=20,
    )


print("Running BDF1-EOP...")
bdf1=run_eop(1)

print("Running BDF2-EOP...")
bdf2=run_eop(2)

print("Running implicit midpoint...")
mid=run_midpoint()


SUBPLOT_MARGINS = dict(
    left=0.22,
    right=0.95,
    bottom=0.20,
    top=0.90,
)
AXES_BOX = [
    SUBPLOT_MARGINS["left"],
    SUBPLOT_MARGINS["bottom"],
    SUBPLOT_MARGINS["right"] - SUBPLOT_MARGINS["left"],
    SUBPLOT_MARGINS["top"] - SUBPLOT_MARGINS["bottom"],
]

# ----------------------------------------------------------------------
# Auxiliary-energy fidelity
# ----------------------------------------------------------------------

defect1 = bdf1.r - (bdf1.H + C_0)
defect2 = bdf2.r - (bdf2.H + C_0)

fig, ax = plt.subplots(figsize=figure_size("single"))

ax.plot(
    bdf1.t,
    defect1,
    "-",
    linewidth=1.4,
    label="BDF-1-EOP",
)
ax.plot(
    bdf2.t,
    defect2,
    "--",
    linewidth=1.4,
    label="BDF-2-EOP",
)

ax.axhline(0.0, linewidth=0.7, alpha=0.35)

ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$r^\ell-(H(x^\ell)+C_0)$")

ax.set_ylim(-1e-14, 1e-14)
ax.set_yticks([-1e-14, -5e-15, 0.0, 5e-15, 1e-14])

ax.grid(True, alpha=0.3)
ax.legend(frameon=False, loc="upper right")

ax.set_position(AXES_BOX)
fig.savefig(OUT / "auxiliary_energy_fidelity.pdf")
plt.close(fig)


# ----------------------------------------------------------------------
# Physical Hamiltonian evolution
# ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=figure_size("single"))

ax.plot(bdf1.t, bdf1.H, "-", label="BDF-1-EOP")
ax.plot(bdf2.t, bdf2.H, "--", label="BDF-2-EOP")
ax.plot(mid.t, mid.H, ":", linewidth=1.8, label="IMP")

ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$H(x^\ell)$")
ax.grid(True, alpha=0.3)
ax.legend(frameon=False)

ax.set_position(AXES_BOX)
fig.savefig(OUT / "hamiltonian_evolution.pdf")
plt.close(fig)

# ----------------------------------------------------------------------
# Save plot data
# ----------------------------------------------------------------------

np.savetxt(
    OUT/"auxiliary_energy_fidelity.csv",
    np.column_stack((bdf1.t,defect1,defect2)),
    delimiter=",",
    header="t,BDF1_defect,BDF2_defect",
    comments="",
)

np.savetxt(
    OUT/"hamiltonian_evolution.csv",
    np.column_stack((bdf1.t,bdf1.H,bdf2.H,mid.H)),
    delimiter=",",
    header="t,BDF1_H,BDF2_H,Midpoint_H",
    comments="",
)


print("\nAuxiliary-energy fidelity")
print(
    f"BDF1-EOP max abs. defect: "
    f"{np.max(np.abs(defect1)):.3e}"
)
print(
    f"BDF2-EOP max abs. defect: "
    f"{np.max(np.abs(defect2)):.3e}"
)

print(f"\nResults written to {OUT}")