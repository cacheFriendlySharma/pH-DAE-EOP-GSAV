from pathlib import Path
import csv

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"benchmark"/"results"/"work_precision"
INPUT=OUT/"work_precision.csv"

STATE_TARGETS=np.array([1e-4,3e-5,1e-5,3e-6])
H_TARGETS=np.array([5e-5,2e-5,1e-5,5e-6,3e-6])

METHODS=("BDF2-EOP","IM-full","IM-modified","IM-simple")


def load_rows():
    rows=[]

    with open(INPUT,newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] not in METHODS:
                continue

            rows.append({
                "method":row["method"],
                "dt":float(row["dt"]),
                "state_error":float(row["state_error"]),
                "H_error":float(row["H_error"]),
                "runtime":float(row["runtime_median"]),
            })

    print(f"Reading measured data from {INPUT}\n")
    return rows


def method_rows(rows,method):
    return [r for r in rows if r["method"]==method]


def log_interp(x,y,target):
    x=np.asarray(x,dtype=float)
    y=np.asarray(y,dtype=float)

    if not (x.min()<=target<=x.max()):
        raise ValueError(
            f"Target {target:.3e} outside measured range "
            f"[{x.min():.3e},{x.max():.3e}]"
        )

    order=np.argsort(x)

    return float(np.exp(
        np.interp(
            np.log(target),
            np.log(x[order]),
            np.log(y[order]),
        )
    ))


def error_to_dt(rows,method,error_key,target):
    data=method_rows(rows,method)

    errors=[r[error_key] for r in data]
    dts=[r["dt"] for r in data]

    return log_interp(errors,dts,target)


def error_to_time(rows,method,error_key,target):
    data=method_rows(rows,method)

    errors=[r[error_key] for r in data]
    times=[r["runtime"] for r in data]

    return log_interp(errors,times,target)


def dt_to_time(rows,method,dt):
    data=method_rows(rows,method)

    dts=[r["dt"] for r in data]
    times=[r["runtime"] for r in data]

    return log_interp(dts,times,dt)


def check_midpoint_equivalence(rows,error_key):
    full={r["dt"]:r[error_key] for r in method_rows(rows,"IM-full")}

    for method in ("IM-modified","IM-simple"):
        max_rel=0.0

        for r in method_rows(rows,method):
            reference=full[r["dt"]]
            rel=abs(r[error_key]-reference)/reference
            max_rel=max(max_rel,rel)

        if max_rel>1e-6:
            raise RuntimeError(
                f"{method} midpoint errors differ unexpectedly from "
                f"IM-full: max relative discrepancy={max_rel:.3e}"
            )

        print(
            f"{method:12s} {error_key} consistency: "
            f"{max_rel:.3e}"
        )


def matched_table(rows,error_key,targets,filename,title):
    check_midpoint_equivalence(rows,error_key)
    print()

    result=[]

    for target in targets:
        # BDF2 accuracy and runtime are interpolated on its own curve.
        dt_bdf2=error_to_dt(
            rows,"BDF2-EOP",error_key,target
        )
        t_bdf2=error_to_time(
            rows,"BDF2-EOP",error_key,target
        )

        # Full Newton defines the common midpoint accuracy curve.
        dt_im=error_to_dt(
            rows,"IM-full",error_key,target
        )

        # All midpoint variants solve the same discrete method.
        # Their cost is therefore evaluated at the same required dt.
        t_full=dt_to_time(rows,"IM-full",dt_im)
        t_modified=dt_to_time(rows,"IM-modified",dt_im)
        t_simple=dt_to_time(rows,"IM-simple",dt_im)

        result.append({
            "target_error":target,
            "BDF2_dt":dt_bdf2,
            "IM_dt":dt_im,
            "BDF2_time":t_bdf2,
            "IM_full_time":t_full,
            "IM_modified_time":t_modified,
            "IM_simple_time":t_simple,
            "speedup_vs_full":t_full/t_bdf2,
            "speedup_vs_modified":t_modified/t_bdf2,
            "speedup_vs_simple":t_simple/t_bdf2,
        })

    print(title)
    print()
    print(
        f"{'target':>10s} {'dt BDF2':>10s} {'dt IM':>10s} "
        f"{'BDF2 [s]':>10s} {'full [s]':>10s} "
        f"{'mod. [s]':>10s} {'simple [s]':>11s} "
        f"{'x full':>8s} {'x mod':>8s} {'x simple':>9s}"
    )
    print("-"*116)

    for r in result:
        print(
            f"{r['target_error']:10.1e} "
            f"{r['BDF2_dt']:10.3e} "
            f"{r['IM_dt']:10.3e} "
            f"{r['BDF2_time']:10.4f} "
            f"{r['IM_full_time']:10.4f} "
            f"{r['IM_modified_time']:10.4f} "
            f"{r['IM_simple_time']:11.4f} "
            f"{r['speedup_vs_full']:8.2f} "
            f"{r['speedup_vs_modified']:8.2f} "
            f"{r['speedup_vs_simple']:9.2f}"
        )

    path=OUT/filename

    with open(path,"w",newline="") as f:
        writer=csv.DictWriter(
            f,
            fieldnames=result[0].keys(),
        )
        writer.writeheader()
        writer.writerows(result)

    print(f"\nWritten to {path}\n")
    return result


rows=load_rows()

matched_table(
    rows,
    error_key="state_error",
    targets=STATE_TARGETS,
    filename="matched_state_accuracy.csv",
    title="Matched state accuracy",
)

matched_table(
    rows,
    error_key="H_error",
    targets=H_TARGETS,
    filename="matched_hamiltonian_accuracy.csv",
    title="Matched Hamiltonian accuracy",
)

print(
    "All reported values are log-log interpolations between measured "
    "work-precision points. No extrapolation is used."
)