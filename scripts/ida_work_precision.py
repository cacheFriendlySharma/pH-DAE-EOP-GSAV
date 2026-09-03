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

from inputs import smooth_pulse,smooth_pulse_average
from potentials import FPUBetaPotential
from system import (
    build_inverse_masses,build_topology,matrix_E,Q_matrix,B_matrix,
    J0_matrix,R0_matrix,Hamiltonian,nonlinear_coenergy,
    g_nonlinear,dissipation,initial_state,physical_strain,
    vector_field,vector_field_jacobian,
)
from time_integrators import eop_gsav


OUT=ROOT/"results"/"ida"
OUT.mkdir(parents=True,exist_ok=True)




# ----------------------------------------------------------------------
# Frozen benchmark
# ----------------------------------------------------------------------

N=8192
neq=4*N-3

m_heavy,m_light=1.0,0.5
beta,beta_J,beta_R=4.0,4.0,2.0
K_M,eta_value=2.0,0.4
a_0,C_0=0.5,1.0
ell_0=ell_f=16
A_u,t_0,tau=2.0,0.5,2.0
T=5.0

audit_dt=2e-2
t_audit=np.arange(0.0,T+0.5*audit_dt,audit_dt)

bdf2_dts=np.array([
    2e-2,
    1e-2,
    5e-3,
    2.5e-3,
    1.25e-3,
])

rtols=np.array([
    1e-3,
    3e-4,
    1e-4,
    3e-5,
    1e-5,
    3e-6,
    1e-6,
])

repeats=3

STATE_TARGETS=np.array([
    1e-4,
    3e-5,
    1e-5,
    3e-6,
])

H_TARGETS=np.array([
    5e-5,
    2e-5,
    1e-5,
    5e-6,
    3e-6,
])


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

potential=FPUBetaPotential(beta=beta)
inverse_masses=build_inverse_masses(N,m_light,m_heavy)
eta_vec=eta_value*np.ones(N-1)
topology=build_topology(N)

E=matrix_E(N)
Q=Q_matrix(N,inverse_masses,K_M,potential)
B=B_matrix(N,ell_f)
Bv=np.asarray(B.toarray()).ravel()

J0=J0_matrix(N,topology)
R0=R0_matrix(N,eta_vec,topology)
x0=initial_state(N,a_0,beta_J,ell_0)

H=lambda x: Hamiltonian(
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

u=lambda t: smooth_pulse(
    t,A_u,t_start=t_0,duration=tau
)

u_bar=lambda a,b: smooth_pulse_average(
    a,b,A_u,t_start=t_0,duration=tau
)


# ----------------------------------------------------------------------
# Local ordering for banded IDA matrix
#
# Original:
#   [p, r, r_M, z]
#
# Local:
#   [p0,r0,rM0,z0,p1,r1,rM1,z1,...,p_{N-1}]
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


# Structural half-bandwidth.
#
# At x0, w=Dv=0, hence
#
#   -3 beta_R D.T diag(w**2) D M^{-1}
#
# vanishes and the instantaneous numerical Jacobian has half-bandwidth 3.
# For a general state w != 0, neighboring momenta couple and the true
# structural half-bandwidth is 4.
half_bandwidth=4

Jtest=Df(x0)[perm,:][:,perm].tocoo()
offsets=Jtest.row-Jtest.col

if np.any(offsets>half_bandwidth) or np.any(offsets<-half_bandwidth):
    raise RuntimeError(
        "Initial permuted Jacobian exceeds structural bandwidth."
    )

initial_bandwidth=int(max(
    np.max(Jtest.row-Jtest.col),
    np.max(Jtest.col-Jtest.row),
))

print(
    f"IDA initial numerical half-bandwidth: {initial_bandwidth}\n"
    f"IDA structural half-bandwidth:        {half_bandwidth}"
)


# ----------------------------------------------------------------------
# Independent DOP853 reference
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

    link=(
        potential.gradient(delta)
        +K_M*r_M
        +beta_R*w**3
    )

    p_dot=np.zeros(N)
    p_dot[:-1]+=link
    p_dot[1:]-=link
    p_dot+=B_p*u(t)[0]

    r_dot=gamma*w
    r_M_dot=w-(K_M/eta_vec)*r_M

    return np.concatenate((
        p_dot,
        r_dot,
        r_M_dot,
    ))


def build_reference():
    y0=x0[:3*N-2]

    sol=solve_ivp(
        reduced_rhs,
        (0.0,T),
        y0,
        method="DOP853",
        t_eval=t_audit,
        rtol=1e-12,
        atol=1e-14,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    p=sol.y[:N].T
    r=sol.y[N:2*N-1].T
    r_M=sol.y[2*N-1:].T

    v=p*inverse_masses
    z=v[:,1:]-(K_M/eta_vec)*r_M

    x=np.concatenate(
        (p,r,r_M,z),
        axis=1,
    )

    H_ref=np.array([H(xi) for xi in x])

    return x,H_ref


def errors(x_num,H_num,x_ref,H_ref):
    Ex=np.max(
        np.linalg.norm(x_num-x_ref,axis=1)
    )
    Ex/=np.max(
        np.linalg.norm(x_ref,axis=1)
    )

    EH=np.max(
        np.abs(H_num-H_ref)
    )
    EH/=np.max(
        np.abs(H_ref)
    )

    return float(Ex),float(EH)


print("\nComputing DOP853 audit reference...")
x_ref,H_ref=build_reference()


# ----------------------------------------------------------------------
# IDA initial data
# ----------------------------------------------------------------------

xdot0=np.zeros(neq)

rhs0=(
    f(x0)
    +Bv*u(0.0)[0]
)

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

        res=(
            np.asarray(E@xdot).ravel()
            -f(x)
            -Bv*u(t)[0]
        )

        rr[:]=to_local(res)

        return 0


    def jacobian(
        self,t,cj,
        yvec,ypvec,rvec,
        J,_,
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
                "IDA Jacobian exceeded declared band structure: "
                f"observed half-bandwidth={actual}, "
                f"declared={half_bandwidth}."
            )

        data=SUNBandMatrix_Data(J)
        ldim=SUNBandMatrix_LDim(J)
        smu=SUNBandMatrix_StoredUpperBandwidth(J)

        data.fill(0.0)

        for i,j,value in zip(
            Jlocal.row,
            Jlocal.col,
            Jlocal.data,
        ):
            data[
                j*ldim+(i-j+smu)
            ]=value

        return 0


def get_stat(fn,mem):
    status,value=fn(mem)

    if status<0:
        raise RuntimeError(
            f"{fn.__name__} failed with status {status}"
        )

    return value


# ----------------------------------------------------------------------
# One complete IDA integration
# ----------------------------------------------------------------------

def run_ida(rtol):
    atol=1e-2*rtol

    problem=IDAProblem()

    status,sunctx=SUNContext_Create(SUN_COMM_NULL)

    if status!=SUN_SUCCESS:
        raise RuntimeError(
            f"SUNContext_Create failed with status {status}"
        )

    y=N_VNew_Serial(neq,sunctx)
    yp=N_VNew_Serial(neq,sunctx)
    idvec=N_VNew_Serial(neq,sunctx)

    N_VGetArrayPointer(y)[:]=y0_local
    N_VGetArrayPointer(yp)[:]=yp0_local
    N_VGetArrayPointer(idvec)[:]=id_local

    ida=IDACreate(sunctx)
    mem=ida.get()

    status=IDAInit(
        mem,
        problem.residual,
        0.0,
        y,
        yp,
    )

    if status<0:
        raise RuntimeError(
            f"IDAInit failed with status {status}"
        )

    status=IDASStolerances(
        mem,
        float(rtol),
        float(atol),
    )

    if status<0:
        raise RuntimeError(
            f"IDASStolerances failed with status {status}"
        )

    status=IDASetId(mem,idvec)

    if status<0:
        raise RuntimeError(
            f"IDASetId failed with status {status}"
        )

    # Unrestricted variable-order IDA:
    # BDF order may increase up to five.
    status=IDASetMaxOrd(mem,5)

    if status<0:
        raise RuntimeError(
            f"IDASetMaxOrd failed with status {status}"
        )

    status=IDASetMaxNumSteps(mem,100000)

    if status<0:
        raise RuntimeError(
            f"IDASetMaxNumSteps failed with status {status}"
        )

    A=SUNBandMatrix(
        neq,
        half_bandwidth,
        half_bandwidth,
        sunctx,
    )

    LS=SUNLinSol_Band(
        y,
        A,
        sunctx,
    )

    status=IDASetLinearSolver(
        mem,
        LS,
        A,
    )

    if status<0:
        raise RuntimeError(
            f"IDASetLinearSolver failed with status {status}"
        )

    status=IDASetJacFn(
        mem,
        problem.jacobian,
    )

    if status<0:
        raise RuntimeError(
            f"IDASetJacFn failed with status {status}"
        )

    x_out=np.empty(
        (len(t_audit),neq)
    )

    x_out[0]=x0

    for k,tout in enumerate(
        t_audit[1:],
        start=1,
    ):
        status,tret=IDASolve(
            mem,
            float(tout),
            y,
            yp,
            IDA_NORMAL,
        )

        if status<0:
            raise RuntimeError(
                f"IDASolve failed at t={tout:.6g}: "
                f"status={status}"
            )

        x_out[k]=to_original(
            N_VGetArrayPointer(y).copy()
        )

    H_out=np.array([
        H(xi) for xi in x_out
    ])

    # Direct algebraic-constraint audit.
    p=x_out[:,:N]
    r_M=x_out[:,2*N-1:3*N-2]
    z=x_out[:,3*N-2:]

    v=p*inverse_masses

    alg=(
        -K_M*r_M
        +eta_vec*(v[:,1:]-z)
    )

    alg_defect=float(
        np.max(np.abs(alg))
    )

    stats={
        "steps":int(
            get_stat(IDAGetNumSteps,mem)
        ),
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

    return (
        x_out,
        H_out,
        alg_defect,
        stats,
    )


# ----------------------------------------------------------------------
# IDA work-precision sweep
# ----------------------------------------------------------------------

print("\nIDA warm-up...")
run_ida(1e-4)

ida_rows=[]
timing_rows=[]

print("\nIDA work-precision sweep\n")

for rtol in rtols:
    times=[]
    representative=None

    print(
        f"rtol={rtol:.1e}",
        end="",
        flush=True,
    )

    for rep in range(1,repeats+1):
        tic=time.perf_counter()

        x_ida,H_ida,alg,stats=run_ida(rtol)

        elapsed=time.perf_counter()-tic

        times.append(elapsed)

        if representative is None:
            representative=(
                x_ida,
                H_ida,
                alg,
                stats,
            )

        timing_rows.append({
            "rtol":rtol,
            "atol":1e-2*rtol,
            "repeat":rep,
            "runtime":elapsed,
        })

        print(
            f"  r{rep}:{elapsed:.4f}s",
            end="",
            flush=True,
        )

    x_ida,H_ida,alg,stats=representative

    Ex,EH=errors(
        x_ida,
        H_ida,
        x_ref,
        H_ref,
    )

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

    ida_rows.append(row)

    print(
        f"  Ex={Ex:.3e}"
        f"  EH={EH:.3e}"
        f"  median={row['runtime_median']:.4f}s"
    )


# ----------------------------------------------------------------------
# Save raw IDA results
# ----------------------------------------------------------------------

with open(
    OUT/"ida_work_precision.csv",
    "w",
    newline="",
) as fcsv:
    writer=csv.DictWriter(
        fcsv,
        fieldnames=ida_rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(ida_rows)


with open(
    OUT/"ida_timing_samples.csv",
    "w",
    newline="",
) as fcsv:
    writer=csv.DictWriter(
        fcsv,
        fieldnames=timing_rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(timing_rows)


# ----------------------------------------------------------------------
# Raw IDA table
# ----------------------------------------------------------------------

print("\nIDA work-precision table\n")

print(
    f"{'rtol':>9s} "
    f"{'Ex':>11s} "
    f"{'EH':>11s} "
    f"{'time [s]':>10s} "
    f"{'steps':>7s} "
    f"{'NLI':>7s} "
    f"{'Jac':>6s} "
    f"{'setup':>7s} "
    f"{'ETF':>5s} "
    f"{'NCF':>5s} "
    f"{'q':>3s}"
)

print("-"*100)

for r in ida_rows:
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
        f"{r['nonlinear_failures']:5d} "
        f"{r['last_order']:3d}"
    )


# ----------------------------------------------------------------------
# Recompute and time BDF2 on exactly the same machine/audit grid
# ----------------------------------------------------------------------

def run_bdf2(dt):
    return eop_gsav(
        J0,R0,B,
        x0,
        0.0,T,
        dt,C_0,
        E=E,
        Q=Q,
        order=2,
        hamiltonian=H,
        nonlinear_coenergy=e_nl,
        g_function=g,
        dissipation_function=K,
        input_average=u_bar,
    )


print("\nBDF2 warm-up...")
run_bdf2(1e-2)

bdf2_rows=[]

print(
    "\nRecomputing BDF2 errors and timings "
    "on the common audit grid...\n"
)

for dt in bdf2_dts:
    times=[]
    representative=None

    print(
        f"dt={dt:.5g}",
        end="",
        flush=True,
    )

    for rep in range(1,repeats+1):
        tic=time.perf_counter()
        result=run_bdf2(dt)
        elapsed=time.perf_counter()-tic

        times.append(elapsed)

        if representative is None:
            representative=result

        print(
            f"  r{rep}:{elapsed:.4f}s",
            end="",
            flush=True,
        )

    result=representative

    idx=np.rint(
        t_audit/dt
    ).astype(int)

    alignment=np.max(
        np.abs(
            result.t[idx]-t_audit
        )
    )

    if alignment>1e-12:
        raise RuntimeError(
            f"Audit grid does not align for dt={dt}: "
            f"error={alignment:.3e}"
        )

    x_num=result.x[idx]
    H_num=result.H[idx]

    Ex,EH=errors(
        x_num,
        H_num,
        x_ref,
        H_ref,
    )

    runtime=float(np.median(times))

    row={
        "dt":float(dt),
        "state_error":Ex,
        "H_error":EH,
        "runtime_median":runtime,
        "runtime_min":float(np.min(times)),
        "runtime_max":float(np.max(times)),
    }

    bdf2_rows.append(row)

    print(
        f"  Ex={Ex:.3e}"
        f"  EH={EH:.3e}"
        f"  median={runtime:.4f}s"
    )


with open(
    OUT/"bdf2_audit.csv",
    "w",
    newline="",
) as fcsv:
    writer=csv.DictWriter(
        fcsv,
        fieldnames=bdf2_rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(bdf2_rows)


# ----------------------------------------------------------------------
# Log-log interpolation
# ----------------------------------------------------------------------

def log_interp(x,y,target):
    x=np.asarray(x,dtype=float)
    y=np.asarray(y,dtype=float)

    if target<x.min() or target>x.max():
        raise ValueError(
            f"Target {target:.3e} outside measured range "
            f"[{x.min():.3e},{x.max():.3e}]"
        )

    order=np.argsort(x)

    return float(
        np.exp(
            np.interp(
                np.log(target),
                np.log(x[order]),
                np.log(y[order]),
            )
        )
    )


def interpolate_method(
    rows,
    error_key,
    target,
    control_key,
):
    errors=[
        r[error_key]
        for r in rows
    ]

    control=log_interp(
        errors,
        [r[control_key] for r in rows],
        target,
    )

    runtime=log_interp(
        errors,
        [r["runtime_median"] for r in rows],
        target,
    )

    return control,runtime


def matched_table(
    error_key,
    targets,
    filename,
    title,
):
    rows=[]

    common_min=max(
        min(r[error_key] for r in bdf2_rows),
        min(r[error_key] for r in ida_rows),
    )

    common_max=min(
        max(r[error_key] for r in bdf2_rows),
        max(r[error_key] for r in ida_rows),
    )

    print(f"\n{title}\n")

    print(
        f"{'target':>10s} "
        f"{'BDF2 dt':>10s} "
        f"{'IDA rtol':>10s} "
        f"{'BDF2 [s]':>10s} "
        f"{'IDA [s]':>10s} "
        f"{'IDA steps':>10s} "
        f"{'IDA/BDF2':>10s}"
    )

    print("-"*84)

    for target in targets:
        if not (
            common_min<=target<=common_max
        ):
            print(
                f"{target:10.1e} "
                f"{'outside common measured range':>70s}"
            )
            continue

        dt_bdf2,t_bdf2=interpolate_method(
            bdf2_rows,
            error_key,
            target,
            "dt",
        )

        rtol_ida,t_ida=interpolate_method(
            ida_rows,
            error_key,
            target,
            "rtol",
        )

        ida_steps=log_interp(
            [r[error_key] for r in ida_rows],
            [r["steps"] for r in ida_rows],
            target,
        )

        ratio=t_ida/t_bdf2

        row={
            "target_error":target,
            "BDF2_dt":dt_bdf2,
            "IDA_rtol":rtol_ida,
            "BDF2_time":t_bdf2,
            "IDA_time":t_ida,
            "IDA_steps":ida_steps,
            "IDA_over_BDF2":ratio,
        }

        rows.append(row)

        print(
            f"{target:10.1e} "
            f"{dt_bdf2:10.3e} "
            f"{rtol_ida:10.3e} "
            f"{t_bdf2:10.4f} "
            f"{t_ida:10.4f} "
            f"{ida_steps:10.1f} "
            f"{ratio:10.2f}"
        )

    if rows:
        with open(
            OUT/filename,
            "w",
            newline="",
        ) as fcsv:
            writer=csv.DictWriter(
                fcsv,
                fieldnames=rows[0].keys(),
            )
            writer.writeheader()
            writer.writerows(rows)

    return rows


# ----------------------------------------------------------------------
# Final matched-accuracy tables
# ----------------------------------------------------------------------

matched_table(
    "state_error",
    STATE_TARGETS,
    "matched_state_accuracy.csv",
    "Matched state accuracy: BDF2-EOP vs IDA",
)


matched_table(
    "H_error",
    H_TARGETS,
    "matched_hamiltonian_accuracy.csv",
    "Matched Hamiltonian accuracy: BDF2-EOP vs IDA",
)


print(f"\nResults written to {OUT}")

print(
    "\nInterpretation: IDA/BDF2 > 1 means BDF2-EOP is faster "
    "at matched observed accuracy; IDA/BDF2 < 1 means IDA is faster."
)
