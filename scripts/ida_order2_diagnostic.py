from pathlib import Path
import csv
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

from sundials4py.core import (
    SUN_COMM_NULL,SUN_SUCCESS,
    SUNContext_Create,
    N_VNew_Serial,N_VGetArrayPointer,
    SUNBandMatrix,SUNBandMatrix_Data,
    SUNBandMatrix_LDim,SUNBandMatrix_StoredUpperBandwidth,
    SUNLinSol_Band,
)
from sundials4py.idas import (
    IDACreate,IDAInit,IDASStolerances,IDASetId,
    IDASetLinearSolver,IDASetJacFn,IDASetMaxOrd,
    IDASetMaxNumSteps,IDASolve,IDA_NORMAL,
    IDAGetNumSteps,IDAGetNumResEvals,
    IDAGetNumLinSolvSetups,IDAGetNumErrTestFails,
    IDAGetNumNonlinSolvIters,IDAGetNumNonlinSolvConvFails,
    IDAGetNumJacEvals,IDAGetLastOrder,
)

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from inputs import smooth_pulse
from potentials import FPUBetaPotential
from system import (
    build_inverse_masses,build_topology,matrix_E,B_matrix,
    Hamiltonian,initial_state,physical_strain,
    vector_field,vector_field_jacobian,
)


OUT=ROOT/"results"/"ida"
OUT.mkdir(parents=True,exist_ok=True)

IDA5_FILE=OUT/"ida_work_precision.csv"
BDF2_FILE=OUT/"bdf2_audit.csv"


# ----------------------------------------------------------------------
# Frozen benchmark
# ----------------------------------------------------------------------

N=8192
neq=4*N-3

m_heavy,m_light=1.0,0.5
beta,beta_J,beta_R=4.0,4.0,2.0
K_M,eta_value=2.0,0.4
a_0=0.5
ell_0=ell_f=16
A_u,t_0,tau=2.0,0.5,2.0
T=5.0

audit_dt=2e-2
t_audit=np.arange(0.0,T+0.5*audit_dt,audit_dt)

rtols=np.array([
    1e-3,3e-4,1e-4,3e-5,1e-5,3e-6,1e-6,
    3e-7,1e-7,3e-8,1e-8,
])

repeats=3
qmax=2


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

potential=FPUBetaPotential(beta=beta)
inverse_masses=build_inverse_masses(N,m_light,m_heavy)
eta_vec=eta_value*np.ones(N-1)
topology=build_topology(N)

E=matrix_E(N)
B=B_matrix(N,ell_f)
Bv=np.asarray(B.toarray()).ravel()
x0=initial_state(N,a_0,beta_J,ell_0)

H=lambda x: Hamiltonian(
    N,inverse_masses,x,potential,K_M,beta_J
)

f=lambda x: vector_field(
    N,inverse_masses,x,potential,K_M,
    eta_vec,beta_J,beta_R
)

Df=lambda x: vector_field_jacobian(
    N,inverse_masses,x,potential,K_M,
    eta_vec,beta_J,beta_R,topology
)

u=lambda t: smooth_pulse(
    t,A_u,t_start=t_0,duration=tau
)


# ----------------------------------------------------------------------
# Local permutation
# ----------------------------------------------------------------------

perm=[]

for i in range(N-1):
    perm.extend((
        i,
        N+i,
        2*N-1+i,
        3*N-2+i,
    ))

perm.append(N-1)
perm=np.asarray(perm,dtype=int)

assert len(perm)==neq
assert np.array_equal(np.sort(perm),np.arange(neq))


def to_local(x):
    return np.asarray(x)[perm]


def to_original(y):
    x=np.empty_like(y)
    x[perm]=y
    return x


# True structural bandwidth; initial numerical bandwidth is only 3
# because w(0)=0.
half_bandwidth=4


# ----------------------------------------------------------------------
# Independent reference
# ----------------------------------------------------------------------

B_p=Bv[:N]


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


def build_reference():
    y0=x0[:3*N-2]

    sol=solve_ivp(
        reduced_rhs,(0.0,T),y0,
        method="DOP853",
        t_eval=t_audit,
        rtol=1e-12,atol=1e-14,
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


def errors(x_num,H_num,x_ref,H_ref):
    Ex=np.max(np.linalg.norm(x_num-x_ref,axis=1))
    Ex/=np.max(np.linalg.norm(x_ref,axis=1))

    EH=np.max(np.abs(H_num-H_ref))
    EH/=np.max(np.abs(H_ref))

    return float(Ex),float(EH)


print("Computing DOP853 audit reference...")
x_ref,H_ref=build_reference()


# ----------------------------------------------------------------------
# IDA initial data
# ----------------------------------------------------------------------

xdot0=np.zeros(neq)
rhs0=f(x0)+Bv*u(0.0)[0]
xdot0[:3*N-2]=rhs0[:3*N-2]

id_original=np.ones(neq)
id_original[3*N-2:]=0.0

y0_local=to_local(x0)
yp0_local=to_local(xdot0)
id_local=to_local(id_original)


# ----------------------------------------------------------------------
# IDA callbacks
# ----------------------------------------------------------------------

class IDAProblem:
    def residual(self,t,yvec,ypvec,rvec,_):
        y=N_VGetArrayPointer(yvec)
        yp=N_VGetArrayPointer(ypvec)
        rr=N_VGetArrayPointer(rvec)

        x=to_original(y)
        xdot=to_original(yp)

        res=np.asarray(E@xdot).ravel()-f(x)-Bv*u(t)[0]
        rr[:]=to_local(res)

        return 0


    def jacobian(
        self,t,cj,yvec,ypvec,rvec,J,_,
        tmp1,tmp2,tmp3,
    ):
        y=N_VGetArrayPointer(yvec)
        x=to_original(y)

        Jorig=cj*E-Df(x)
        Jlocal=Jorig[perm,:][:,perm].tocoo()
        Jlocal.sum_duplicates()

        offsets=Jlocal.row-Jlocal.col

        if (
            np.any(offsets>half_bandwidth)
            or np.any(offsets<-half_bandwidth)
        ):
            actual=int(np.max(np.abs(offsets)))
            raise RuntimeError(
                f"Observed half-bandwidth={actual}, "
                f"declared={half_bandwidth}."
            )

        data=SUNBandMatrix_Data(J)
        ldim=SUNBandMatrix_LDim(J)
        smu=SUNBandMatrix_StoredUpperBandwidth(J)

        data.fill(0.0)

        for i,j,value in zip(
            Jlocal.row,Jlocal.col,Jlocal.data
        ):
            data[j*ldim+(i-j+smu)]=value

        return 0


def get_stat(fn,mem):
    status,value=fn(mem)

    if status<0:
        raise RuntimeError(
            f"{fn.__name__} failed with status {status}"
        )

    return value


# ----------------------------------------------------------------------
# IDA q <= 2 integration
# ----------------------------------------------------------------------

def run_ida(rtol):
    atol=1e-2*rtol
    problem=IDAProblem()

    status,sunctx=SUNContext_Create(SUN_COMM_NULL)

    if status!=SUN_SUCCESS:
        raise RuntimeError("SUNContext_Create failed.")

    y=N_VNew_Serial(neq,sunctx)
    yp=N_VNew_Serial(neq,sunctx)
    idvec=N_VNew_Serial(neq,sunctx)

    N_VGetArrayPointer(y)[:]=y0_local
    N_VGetArrayPointer(yp)[:]=yp0_local
    N_VGetArrayPointer(idvec)[:]=id_local

    ida=IDACreate(sunctx)
    mem=ida.get()

    if IDAInit(mem,problem.residual,0.0,y,yp)<0:
        raise RuntimeError("IDAInit failed.")

    if IDASStolerances(mem,float(rtol),float(atol))<0:
        raise RuntimeError("IDASStolerances failed.")

    if IDASetId(mem,idvec)<0:
        raise RuntimeError("IDASetId failed.")

    if IDASetMaxOrd(mem,qmax)<0:
        raise RuntimeError("IDASetMaxOrd failed.")

    if IDASetMaxNumSteps(mem,100000)<0:
        raise RuntimeError("IDASetMaxNumSteps failed.")

    A=SUNBandMatrix(
        neq,half_bandwidth,half_bandwidth,sunctx
    )
    LS=SUNLinSol_Band(y,A,sunctx)

    if IDASetLinearSolver(mem,LS,A)<0:
        raise RuntimeError("IDASetLinearSolver failed.")

    if IDASetJacFn(mem,problem.jacobian)<0:
        raise RuntimeError("IDASetJacFn failed.")

    x_out=np.empty((len(t_audit),neq))
    x_out[0]=x0

    for k,tout in enumerate(t_audit[1:],1):
        status,tret=IDASolve(
            mem,float(tout),y,yp,IDA_NORMAL
        )

        if status<0:
            raise RuntimeError(
                f"IDASolve failed at t={tout}: {status}"
            )

        x_out[k]=to_original(
            N_VGetArrayPointer(y).copy()
        )

    H_out=np.array([H(xi) for xi in x_out])

    p=x_out[:,:N]
    r_M=x_out[:,2*N-1:3*N-2]
    z=x_out[:,3*N-2:]
    v=p*inverse_masses

    alg=-K_M*r_M+eta_vec*(v[:,1:]-z)
    alg_defect=float(np.max(np.abs(alg)))

    stats={
        "steps":int(get_stat(IDAGetNumSteps,mem)),
        "residual_evaluations":int(
            get_stat(IDAGetNumResEvals,mem)
        ),
        "jacobian_evaluations":int(
            get_stat(IDAGetNumJacEvals,mem)
        ),
        "linear_setups":int(
            get_stat(IDAGetNumLinSolvSetups,mem)
        ),
        "nonlinear_iterations":int(
            get_stat(IDAGetNumNonlinSolvIters,mem)
        ),
        "error_test_failures":int(
            get_stat(IDAGetNumErrTestFails,mem)
        ),
        "nonlinear_failures":int(
            get_stat(IDAGetNumNonlinSolvConvFails,mem)
        ),
        "last_order":int(
            get_stat(IDAGetLastOrder,mem)
        ),
    }

    return x_out,H_out,alg_defect,stats


# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------

print("\nIDA q_max=2 warm-up...")
run_ida(1e-4)

rows=[]
timing=[]

print("\nIDA q_max=2 work-precision sweep\n")

for rtol in rtols:
    times=[]
    representative=None

    print(f"rtol={rtol:.1e}",end="",flush=True)

    for rep in range(1,repeats+1):
        tic=time.perf_counter()
        x,H_num,alg,stats=run_ida(rtol)
        elapsed=time.perf_counter()-tic

        times.append(elapsed)

        if representative is None:
            representative=(x,H_num,alg,stats)

        timing.append({
            "rtol":rtol,
            "atol":1e-2*rtol,
            "repeat":rep,
            "runtime":elapsed,
        })

        print(f"  r{rep}:{elapsed:.4f}s",end="",flush=True)

    x,H_num,alg,stats=representative
    Ex,EH=errors(x,H_num,x_ref,H_ref)

    row={
        "rtol":rtol,
        "atol":1e-2*rtol,
        "state_error":Ex,
        "H_error":EH,
        "runtime_median":float(np.median(times)),
        "runtime_min":float(np.min(times)),
        "runtime_max":float(np.max(times)),
        "algebraic_defect":alg,
        **stats,
    }
    rows.append(row)

    print(
        f"  Ex={Ex:.3e}"
        f"  EH={EH:.3e}"
        f"  median={row['runtime_median']:.4f}s"
    )


with open(
    OUT/"ida_order2_work_precision.csv","w",newline=""
) as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


with open(
    OUT/"ida_order2_timing_samples.csv","w",newline=""
) as f:
    writer=csv.DictWriter(f,fieldnames=timing[0].keys())
    writer.writeheader()
    writer.writerows(timing)


# ----------------------------------------------------------------------
# Raw q <= 2 table
# ----------------------------------------------------------------------

print("\nIDA q_max=2 work-precision table\n")
print(
    f"{'rtol':>9s} {'Ex':>11s} {'EH':>11s} {'time [s]':>10s} "
    f"{'steps':>7s} {'NLI':>7s} {'Jac':>6s} {'setup':>7s} "
    f"{'ETF':>5s} {'q':>3s}"
)
print("-"*93)

for r in rows:
    print(
        f"{r['rtol']:9.1e} "
        f"{r['state_error']:11.3e} "
        f"{r['H_error']:11.3e} "
        f"{r['runtime_median']:10.4f} "
        f"{r['steps']:7d} "
        f"{r['nonlinear_iterations']:7d} "
        f"{r['jacobian_evaluations']:6d} "
        f"{r['linear_setups']:7d} "
        f"{r['error_test_failures']:5d} "
        f"{r['last_order']:3d}"
    )


# ----------------------------------------------------------------------
# Load unrestricted IDA and BDF2 measurements
# ----------------------------------------------------------------------

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)

    result=[]

    with open(path,newline="") as f:
        for row in csv.DictReader(f):
            result.append({
                key: (
                    value
                    if key=="method"
                    else float(value)
                )
                for key,value in row.items()
            })

    return result


ida5=load_csv(IDA5_FILE)
bdf2=load_csv(BDF2_FILE)


# ----------------------------------------------------------------------
# Same-tolerance q2 versus q5
# ----------------------------------------------------------------------

print("\nIDA order restriction diagnostic: same tolerance\n")
print(
    f"{'rtol':>9s} "
    f"{'Ex q2':>11s} {'Ex q5':>11s} "
    f"{'t q2':>9s} {'t q5':>9s} "
    f"{'steps q2':>9s} {'steps q5':>9s} "
    f"{'q2/q5':>8s}"
)
print("-"*92)

for q2 in rows:
    matches=[
        r for r in ida5
        if np.isclose(r["rtol"],q2["rtol"],rtol=1e-12,atol=0.0)
    ]
    if not matches:
        continue

    q5=matches[0]
    
    print(
        f"{q2['rtol']:9.1e} "
        f"{q2['state_error']:11.3e} "
        f"{q5['state_error']:11.3e} "
        f"{q2['runtime_median']:9.4f} "
        f"{q5['runtime_median']:9.4f} "
        f"{q2['steps']:9.0f} "
        f"{q5['steps']:9.0f} "
        f"{q2['runtime_median']/q5['runtime_median']:8.2f}"
    )


# ----------------------------------------------------------------------
# Fastest actually measured IDA configuration satisfying an accuracy
# requirement. No interpolation, no assumed monotonic tolerance curve.
# ----------------------------------------------------------------------

def best_measured(data,error_key,target):
    candidates=[
        r for r in data
        if r[error_key]<=target
    ]

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda r:r["runtime_median"]
    )


def dominance_table(error_key,title):
    print(f"\n{title}\n")
    print(
        f"{'BDF2 err':>11s} {'BDF2 [s]':>10s} "
        f"{'q5 err':>11s} {'q5 [s]':>9s} {'q5 speed':>9s} "
        f"{'q2 err':>11s} {'q2 [s]':>9s} {'q2 speed':>9s}"
    )
    print("-"*102)

    output=[]

    for b in sorted(
        bdf2,
        key=lambda r:r[error_key],
        reverse=True,
    ):
        target=b[error_key]

        q5=best_measured(
            ida5,error_key,target
        )
        q2=best_measured(
            rows,error_key,target
        )

        if q5 is None or q2 is None:
            continue

        speed5=b["runtime_median"]/q5["runtime_median"]
        speed2=b["runtime_median"]/q2["runtime_median"]

        print(
            f"{target:11.3e} "
            f"{b['runtime_median']:10.4f} "
            f"{q5[error_key]:11.3e} "
            f"{q5['runtime_median']:9.4f} "
            f"{speed5:9.2f} "
            f"{q2[error_key]:11.3e} "
            f"{q2['runtime_median']:9.4f} "
            f"{speed2:9.2f}"
        )

        output.append({
            "BDF2_error":target,
            "BDF2_time":b["runtime_median"],
            "IDA5_error":q5[error_key],
            "IDA5_rtol":q5["rtol"],
            "IDA5_time":q5["runtime_median"],
            "BDF2_over_IDA5":speed5,
            "IDA2_error":q2[error_key],
            "IDA2_rtol":q2["rtol"],
            "IDA2_time":q2["runtime_median"],
            "BDF2_over_IDA2":speed2,
        })

    return output


state=dominance_table(
    "state_error",
    "Measured state-accuracy comparison",
)

energy=dominance_table(
    "H_error",
    "Measured Hamiltonian-accuracy comparison",
)


for filename,data in (
    ("measured_state_comparison.csv",state),
    ("measured_hamiltonian_comparison.csv",energy),
):
    if data:
        with open(OUT/filename,"w",newline="") as f:
            writer=csv.DictWriter(
                f,fieldnames=data[0].keys()
            )
            writer.writeheader()
            writer.writerows(data)


print(f"\nResults written to {OUT}")

print(
    "\nThe q_max=5 configuration remains the primary IDA baseline. "
    "The q_max=2 results are diagnostic only."
)
