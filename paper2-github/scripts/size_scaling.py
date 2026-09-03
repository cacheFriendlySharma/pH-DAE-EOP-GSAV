from pathlib import Path
import csv
import os
import sys
import time

# Keep the laptop scaling experiment effectively single-threaded unless the
# environment has already been configured explicitly.
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from inputs import smooth_pulse,smooth_pulse_average
from potentials import FPUBetaPotential
from system import (
    build_inverse_masses,build_topology,matrix_E,Q_matrix,B_matrix,
    J0_matrix,R0_matrix,Hamiltonian,coenergy,nonlinear_coenergy,
    g_nonlinear,dissipation,initial_state,vector_field,vector_field_jacobian,
)
from time_integrators import eop_gsav
from implicit_midpoint import implicit_midpoint
from newton_solvers import full_newton,modified_newton,simple_newton


OUT=ROOT/"benchmark"/"results"/"size_scaling"
OUT.mkdir(parents=True,exist_ok=True)

SAMPLES=OUT/"timing_samples.csv"
SUMMARY=OUT/"size_scaling.csv"


# ----------------------------------------------------------------------
# Frozen benchmark
# ----------------------------------------------------------------------

Ns = (64, 128, 256, 512, 1024, 2048, 4096,
      8192, 16384, 32768)

dt=5e-3
T=5.0
repeats=3

m_heavy,m_light=1.0,0.5
beta,beta_J,beta_R=4.0,4.0,2.0
K_M,eta_value=2.0,0.4
a_0,C_0=0.5,1.0
ell_0=ell_f=16
A_u,t_0,tau=2.0,0.5,2.0

newton_tol=1e-11
newton_maxiter=30

METHODS=(
    "BDF1-EOP",
    "BDF2-EOP",
    "IM-full",
    "IM-modified",
    "IM-simple",
)

SAMPLE_FIELDS=[
    "N","unknowns","method","repeat","runtime","steps",
    "converged","mean_newton","max_newton",
    "residual_evaluations","jacobian_evaluations",
    "factorizations","linear_solves","H_final_over_H0",
]

SUMMARY_FIELDS=[
    "N","unknowns","method","steps",
    "runtime_median","runtime_min","runtime_max",
    "time_per_step",
    "converged","mean_newton","max_newton",
    "residual_evaluations","jacobian_evaluations",
    "factorizations","linear_solves","H_final_over_H0",
    "speedup_vs_full","speedup_vs_simple",
]


# ----------------------------------------------------------------------
# Model factory
# ----------------------------------------------------------------------

def build_model(N):
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
    u=lambda t: smooth_pulse(
        t,A_u,t_start=t_0,duration=tau
    )
    u_bar=lambda a,b: smooth_pulse_average(
        a,b,A_u,t_start=t_0,duration=tau
    )

    return {
        "E":E,"Q":Q,"B":B,"J0":J0,"R0":R0,"x0":x0,
        "H":H,"e":e,"e_nl":e_nl,"g":g,"K":K,
        "f":f,"Df":Df,"u":u,"u_bar":u_bar,
    }


def solve(model,method,T_end=T):
    if method=="BDF1-EOP":
        return eop_gsav(
            model["J0"],model["R0"],model["B"],
            model["x0"],0.0,T_end,dt,C_0,
            E=model["E"],Q=model["Q"],order=1,
            hamiltonian=model["H"],
            nonlinear_coenergy=model["e_nl"],
            g_function=model["g"],
            dissipation_function=model["K"],
            input_average=model["u_bar"],
        )

    if method=="BDF2-EOP":
        return eop_gsav(
            model["J0"],model["R0"],model["B"],
            model["x0"],0.0,T_end,dt,C_0,
            E=model["E"],Q=model["Q"],order=2,
            hamiltonian=model["H"],
            nonlinear_coenergy=model["e_nl"],
            g_function=model["g"],
            dissipation_function=model["K"],
            input_average=model["u_bar"],
        )

    solver={
        "IM-full":full_newton,
        "IM-modified":modified_newton,
        "IM-simple":simple_newton,
    }[method]

    return implicit_midpoint(
        model["E"],model["B"],model["x0"],
        0.0,T_end,dt,solver,
        hamiltonian=model["H"],
        effort_function=model["e"],
        vector_field=model["f"],
        vector_field_jacobian=model["Df"],
        dissipation_function=model["K"],
        input_function=model["u"],
        newton_tol=newton_tol,
        newton_maxiter=newton_maxiter,
        use_frozen_jacobian=(method=="IM-simple"),
    )


# ----------------------------------------------------------------------
# Persistent checkpointing
# ----------------------------------------------------------------------

def load_samples():
    if not SAMPLES.exists():
        return []

    rows=[]
    with open(SAMPLES,newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "N":int(row["N"]),
                "unknowns":int(row["unknowns"]),
                "method":row["method"],
                "repeat":int(row["repeat"]),
                "runtime":float(row["runtime"]),
                "steps":int(row["steps"]),
                "converged":row["converged"]=="True",
                "mean_newton":float(row["mean_newton"]),
                "max_newton":int(row["max_newton"]),
                "residual_evaluations":int(row["residual_evaluations"]),
                "jacobian_evaluations":int(row["jacobian_evaluations"]),
                "factorizations":int(row["factorizations"]),
                "linear_solves":int(row["linear_solves"]),
                "H_final_over_H0":float(row["H_final_over_H0"]),
            })
    return rows


def append_sample(row):
    exists=SAMPLES.exists()

    with open(SAMPLES,"a",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=SAMPLE_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def completed_repeats(samples,N,method):
    return {
        r["repeat"] for r in samples
        if r["N"]==N and r["method"]==method
    }


def collect_metadata(result,method,H0):
    steps=len(result.t)-1

    if method.startswith("IM"):
        return {
            "steps":steps,
            "converged":bool(np.asarray(result.converged).all()),
            "mean_newton":float(np.mean(result.newton_iterations)),
            "max_newton":int(np.max(result.newton_iterations)),
            "residual_evaluations":int(result.residual_evaluations),
            "jacobian_evaluations":int(result.jacobian_evaluations),
            "factorizations":int(result.factorizations),
            "linear_solves":int(result.linear_solves),
            "H_final_over_H0":float(result.H[-1]/H0),
        }

    return {
        "steps":steps,
        "converged":True,
        "mean_newton":np.nan,
        "max_newton":-1,
        "residual_evaluations":-1,
        "jacobian_evaluations":-1,
        "factorizations":1 if method=="BDF1-EOP" else 2,
        "linear_solves":steps,
        "H_final_over_H0":float(result.H[-1]/H0),
    }


# ----------------------------------------------------------------------
# Summary construction
# ----------------------------------------------------------------------

def build_summary(samples):
    rows=[]

    for N in Ns:
        for method in METHODS:
            data=[
                r for r in samples
                if r["N"]==N and r["method"]==method
            ]

            if len(data)<repeats:
                continue

            data=sorted(data,key=lambda r:r["repeat"])
            times=np.array([r["runtime"] for r in data])
            meta=data[0]

            rows.append({
                "N":N,
                "unknowns":4*N-3,
                "method":method,
                "steps":meta["steps"],
                "runtime_median":float(np.median(times)),
                "runtime_min":float(np.min(times)),
                "runtime_max":float(np.max(times)),
                "time_per_step":float(np.median(times)/meta["steps"]),
                "converged":all(r["converged"] for r in data),
                "mean_newton":meta["mean_newton"],
                "max_newton":meta["max_newton"],
                "residual_evaluations":meta["residual_evaluations"],
                "jacobian_evaluations":meta["jacobian_evaluations"],
                "factorizations":meta["factorizations"],
                "linear_solves":meta["linear_solves"],
                "H_final_over_H0":meta["H_final_over_H0"],
                "speedup_vs_full":np.nan,
                "speedup_vs_simple":np.nan,
            })

    for row in rows:
        full=next(
            (
                r for r in rows
                if r["N"]==row["N"] and r["method"]=="IM-full"
            ),
            None,
        )
        simple=next(
            (
                r for r in rows
                if r["N"]==row["N"] and r["method"]=="IM-simple"
            ),
            None,
        )

        if full is not None:
            row["speedup_vs_full"]=(
                full["runtime_median"]/row["runtime_median"]
            )

        if simple is not None:
            row["speedup_vs_simple"]=(
                simple["runtime_median"]/row["runtime_median"]
            )

    with open(SUMMARY,"w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def print_tables(rows):
    print("\nSize-scaling runtime\n")
    print(
        f"{'N':>5s} {'n_x':>6s} {'method':>12s} "
        f"{'time [s]':>10s} {'ms/step':>10s} "
        f"{'x full':>8s} {'x simple':>9s}"
    )
    print("-"*73)

    for N in Ns:
        for r in rows:
            if r["N"]!=N:
                continue

            sf=(
                "-"
                if np.isnan(r["speedup_vs_full"])
                else f"{r['speedup_vs_full']:.2f}"
            )
            ss=(
                "-"
                if np.isnan(r["speedup_vs_simple"])
                else f"{r['speedup_vs_simple']:.2f}"
            )

            print(
                f"{N:5d} {r['unknowns']:6d} {r['method']:>12s} "
                f"{r['runtime_median']:10.4f} "
                f"{1e3*r['time_per_step']:10.4f} "
                f"{sf:>8s} {ss:>9s}"
            )

    print("\nSolver-cost anatomy\n")
    print(
        f"{'N':>5s} {'method':>12s} {'mean N':>8s} "
        f"{'max N':>7s} {'Jac':>8s} {'LU':>8s} {'solves':>9s}"
    )
    print("-"*68)

    for N in Ns:
        for r in rows:
            if r["N"]!=N:
                continue

            if r["method"].startswith("IM"):
                meanN=f"{r['mean_newton']:.2f}"
                maxN=f"{r['max_newton']}"
                jac=f"{r['jacobian_evaluations']}"
            else:
                meanN=maxN=jac="-"

            print(
                f"{N:5d} {r['method']:>12s} "
                f"{meanN:>8s} {maxN:>7s} {jac:>8s} "
                f"{r['factorizations']:8d} "
                f"{r['linear_solves']:9d}"
            )

    print("\nEmpirical runtime scaling exponent")
    print(r"p = log(t_2/t_1) / log(N_2/N_1)")
    print()
    print(
        f"{'method':>12s} "
        + " ".join(
            f"{Ns[j]}->{Ns[j+1]:<4d}"
            for j in range(len(Ns)-1)
        )
    )
    print("-"*55)

    for method in METHODS:
        values=[]
        valid=True

        for N in Ns:
            row=next(
                (
                    r for r in rows
                    if r["N"]==N and r["method"]==method
                ),
                None,
            )

            if row is None:
                valid=False
                break
            values.append(row["runtime_median"])

        if not valid:
            continue

        p=[
            np.log(values[j+1]/values[j])
            /np.log(Ns[j+1]/Ns[j])
            for j in range(len(Ns)-1)
        ]

        print(
            f"{method:>12s} "
            +" ".join(f"{value:10.3f}" for value in p)
        )


# ----------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------

def main():
    samples=load_samples()

    if samples:
        print(
            f"Resuming from {len(samples)} existing timing samples "
            f"in {SAMPLES}"
        )

    for N in Ns:
        print(f"\nN={N}, unknowns={4*N-3}")
        model=build_model(N)
        H0=model["H"](model["x0"])

        for method in METHODS:
            done=completed_repeats(samples,N,method)

            if len(done)>=repeats:
                print(f"  {method:12s} complete")
                continue

            print(f"  {method:12s}",end="",flush=True)

            # Short warm-up only if this configuration has no saved run yet.
            if not done:
                warm_steps=5
                solve(model,method,T_end=warm_steps*dt)

            for rep in range(1,repeats+1):
                if rep in done:
                    print(f" r{rep}:saved",end="",flush=True)
                    continue

                tic=time.perf_counter()
                result=solve(model,method,T)
                elapsed=time.perf_counter()-tic

                meta=collect_metadata(result,method,H0)

                row={
                    "N":N,
                    "unknowns":4*N-3,
                    "method":method,
                    "repeat":rep,
                    "runtime":elapsed,
                    **meta,
                }

                append_sample(row)
                samples.append(row)

                print(f" r{rep}:{elapsed:.3f}s",end="",flush=True)

            times=[
                r["runtime"] for r in samples
                if r["N"]==N and r["method"]==method
            ]
            print(f" median={np.median(times):.4f}s")

            # Rebuild the summary after every completed configuration.
            build_summary(samples)

    rows=build_summary(samples)
    print_tables(rows)

    print(f"\nRaw timings: {SAMPLES}")
    print(f"Summary:     {SUMMARY}")


if __name__=="__main__":
    main()