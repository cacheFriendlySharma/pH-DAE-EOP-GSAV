from pathlib import Path
import csv,sys
import numpy as np

from sundials4py.core import (
    SUN_COMM_NULL,SUNContext_Create,N_VNew_Serial,N_VGetArrayPointer,
    SUNBandMatrix,SUNBandMatrix_Data,SUNBandMatrix_LDim,
    SUNBandMatrix_StoredUpperBandwidth,SUNLinSol_Band,
)
from sundials4py.idas import (
    IDACreate,IDAInit,IDASStolerances,IDASetId,IDASetLinearSolver,
    IDASetJacFn,IDASetMaxOrd,IDASetMaxNumSteps,IDASolve,IDA_NORMAL,
    IDAGetNumLinSolvSetups,
)

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from inputs import smooth_pulse,smooth_pulse_average
from potentials import FPUBetaPotential
from system import (
    build_inverse_masses,build_topology,matrix_E,Q_matrix,B_matrix,
    J0_matrix,R0_matrix,Hamiltonian,nonlinear_coenergy,g_nonlinear,
    dissipation,initial_state,physical_strain,vector_field,
    vector_field_jacobian,
)
from time_integrators import eop_gsav


# benchmark
N,dt,T=64,1e-2,5.0
beta,beta_J,beta_R=4.0,4.0,2.0
K_M,eta=2.0,0.4
m_light,m_heavy=0.5,1.0
a0,C0,ell0,ellf=0.5,1.0,16,16
Au,t0,tau=2.0,0.5,2.0
rtol,atol=1e-5,1e-7
n=4*N-3

potential=FPUBetaPotential(beta=beta)
Minv=build_inverse_masses(N,m_light,m_heavy)
eta_v=eta*np.ones(N-1)
top=build_topology(N)
E=matrix_E(N)
Q=Q_matrix(N,Minv,K_M,potential)
B=B_matrix(N,ellf)
Bv=np.asarray(B.toarray()).ravel()
J0,R0=J0_matrix(N,top),R0_matrix(N,eta_v,top)
x0=initial_state(N,a0,beta_J,ell0)

H=lambda x: Hamiltonian(N,Minv,x,potential,K_M,beta_J)
enl=lambda x: nonlinear_coenergy(N,x,potential,beta_J)
g=lambda x: g_nonlinear(N,Minv,x,potential,beta_J,beta_R)
K=lambda x: dissipation(N,Minv,x,eta_v,beta_R)
f=lambda x: vector_field(N,Minv,x,potential,K_M,eta_v,beta_J,beta_R)
Df=lambda x: vector_field_jacobian(
    N,Minv,x,potential,K_M,eta_v,beta_J,beta_R,top
)
u=lambda t: smooth_pulse(t,Au,t_start=t0,duration=tau)
ubar=lambda a,b: smooth_pulse_average(a,b,Au,t_start=t0,duration=tau)


def effort(x):
    p=x[:N]
    r=x[N:2*N-1]
    rM=x[2*N-1:3*N-2]
    z=x[3*N-2:]
    delta=physical_strain(r,beta_J)
    gamma=np.sqrt(1.0+beta_J*r*r)
    return np.concatenate((
        Minv*p,
        (delta+beta*delta**3)/gamma,
        K_M*rM,
        z,
    ))


def power(t,x):
    return float(u(t)[0]*(B.T@effort(x))[0])


def alg_defect(X):
    p=X[:,:N]
    rM=X[:,2*N-1:3*N-2]
    z=X[:,3*N-2:]
    v=p*Minv
    return np.max(np.abs(-K_M*rM+eta_v*(v[:,1:]-z)))


# BDF2-EOP-GSAV
gs=eop_gsav(
    J0,R0,B,x0,0.0,T,dt,C0,E=E,Q=Q,order=2,
    hamiltonian=H,nonlinear_coenergy=enl,g_function=g,
    dissipation_function=K,input_average=ubar,
)

native=np.max(np.maximum(np.diff(gs.r)-dt*gs.power,0.0))
tracking=np.max(np.abs(gs.r-(gs.H+C0)))


# IDA local permutation
perm=[]
for i in range(N-1):
    perm.extend((i,N+i,2*N-1+i,3*N-2+i))
perm=np.asarray(perm+[N-1])

def local(x): return np.asarray(x)[perm]

def original(y):
    x=np.empty_like(y)
    x[perm]=y
    return x


class Problem:
    def residual(self,t,yv,ypv,rv,_):
        y,yp=N_VGetArrayPointer(yv),N_VGetArrayPointer(ypv)
        N_VGetArrayPointer(rv)[:]=local(
            np.asarray(E@original(yp)).ravel()-f(original(y))-Bv*u(t)[0]
        )
        return 0

    def jacobian(self,t,cj,yv,ypv,rv,J,_,tmp1,tmp2,tmp3):
        A=(cj*E-Df(original(N_VGetArrayPointer(yv))))[perm,:][:,perm].tocoo()
        A.sum_duplicates()
        data=SUNBandMatrix_Data(J)
        data.fill(0.0)
        ld,smu=SUNBandMatrix_LDim(J),SUNBandMatrix_StoredUpperBandwidth(J)
        for i,j,v in zip(A.row,A.col,A.data):
            data[j*ld+i-j+smu]=v
        return 0


status,ctx=SUNContext_Create(SUN_COMM_NULL)
y=N_VNew_Serial(n,ctx)
yp=N_VNew_Serial(n,ctx)
idv=N_VNew_Serial(n,ctx)

xdot0=np.zeros(n)
xdot0[:3*N-2]=(f(x0)+Bv*u(0.0)[0])[:3*N-2]
ids=np.ones(n)
ids[3*N-2:]=0.0

N_VGetArrayPointer(y)[:]=local(x0)
N_VGetArrayPointer(yp)[:]=local(xdot0)
N_VGetArrayPointer(idv)[:]=local(ids)

problem=Problem()
ida=IDACreate(ctx)
mem=ida.get()

IDAInit(mem,problem.residual,0.0,y,yp)
IDASStolerances(mem,rtol,atol)
IDASetId(mem,idv)
IDASetMaxOrd(mem,5)
IDASetMaxNumSteps(mem,100000)

A=SUNBandMatrix(n,4,4,ctx)
LS=SUNLinSol_Band(y,A,ctx)
IDASetLinearSolver(mem,LS,A)
IDASetJacFn(mem,problem.jacobian)

t=np.arange(0.0,T+0.5*dt,dt)
X=np.empty((len(t),n))
X[0]=x0

for k,tk in enumerate(t[1:],1):
    flag,_=IDASolve(mem,float(tk),y,yp,IDA_NORMAL)
    if flag<0:
        raise RuntimeError(f"IDA failed at t={tk}: {flag}")
    X[k]=original(N_VGetArrayPointer(y).copy())

Hida=np.array([H(x) for x in X])


# Common physical audit: same grid and same trapezoidal P,K quadrature
def physical_audit(t,X,Hx):
    P=np.array([power(ti,xi) for ti,xi in zip(t,X)])
    D=np.array([K(xi) for xi in X])
    W=.5*dt*(P[:-1]+P[1:])
    Diss=.5*dt*(D[:-1]+D[1:])
    dH=np.diff(Hx)
    return (
        np.max(np.maximum(dH-W,0.0)),
        np.max(np.abs(dH-W+Diss)),
    )


gpass,gbal=physical_audit(gs.t,gs.x,gs.H)
ipass,ibal=physical_audit(t,X,Hida)

flag,setups=IDAGetNumLinSolvSetups(mem)
if flag<0:
    raise RuntimeError("IDAGetNumLinSolvSetups failed")

rows=[
    {
        "method":"BDF2-EOP-GSAV",
        "native_passivity_violation":native,
        "physical_passivity_violation":gpass,
        "physical_balance_defect":gbal,
        "algebraic_defect":alg_defect(gs.x),
        "auxiliary_energy_gap":tracking,
        "linear_setups":2,
    },
    {
        "method":"IDA",
        "native_passivity_violation":np.nan,
        "physical_passivity_violation":ipass,
        "physical_balance_defect":ibal,
        "algebraic_defect":alg_defect(X),
        "auxiliary_energy_gap":np.nan,
        "linear_setups":int(setups),
    },
]

print("\nStructure comparison, N=64\n")
print(
    f"{'method':18s} {'native viol.':>13s} {'phys. viol.':>13s} "
    f"{'balance':>12s} {'alg. defect':>12s} {'|r-H-C0|':>12s} {'setups':>7s}"
)
print("-"*96)

for r in rows:
    def fmt(x): return "—" if np.isnan(x) else f"{x:.3e}"
    print(
        f"{r['method']:18s} {fmt(r['native_passivity_violation']):>13s} "
        f"{r['physical_passivity_violation']:13.3e} "
        f"{r['physical_balance_defect']:12.3e} "
        f"{r['algebraic_defect']:12.3e} "
        f"{fmt(r['auxiliary_energy_gap']):>12s} "
        f"{r['linear_setups']:7d}"
    )

out=ROOT/"benchmark"/"results"/"ida"
out.mkdir(parents=True,exist_ok=True)
with open(out/"ida_gsav_structure.csv","w",newline="") as fcsv:
    writer=csv.DictWriter(fcsv,fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nStructural guarantees")
print("BDF2-EOP-GSAV : discrete auxiliary passivity = YES; constant matrices = YES")
print("IDA            : discrete pH passivity guarantee = NO; constant matrices = NO")