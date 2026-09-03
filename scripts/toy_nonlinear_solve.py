from pathlib import Path
import csv
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import lu_factor, lu_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inputs import smooth_pulse, smooth_pulse_average


OUT = ROOT / "benchmark" / "results" / "toy_nonlinear_solve"
OUT.mkdir(parents=True, exist_ok=True)



# ----------------------------------------------------------------------
# Frozen problem
# ----------------------------------------------------------------------

T = 5.0

omega = c = rho = 1.0
nu = gamma = 0.05
mu = 0.5
kappa = 1.0
beta = 32.0

A_u = 2.0
t0 = 0.5
tau = 2.0
C0 = 1.0

dts = [0.1, 0.05, 0.025, 0.0125, 0.00625]
repeats = 3

newton_tol = 1e-10
newton_maxiter = 20


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

E = np.diag([1.0, 1.0, 0.0])
B = np.array([0.0, 1.0, 0.0])

J0 = np.array([
    [0.0, omega, 0.0],
    [-omega, 0.0, -c],
    [0.0, c, 0.0],
])

R0 = np.diag([nu, nu, rho])
A = R0 - J0

x0 = np.zeros(3)


def u(t):
    return float(np.asarray(
        smooth_pulse(t, A_u, t_start=t0, duration=tau)
    ).item())


def ubar(ta, tb):
    return float(np.asarray(
        smooth_pulse_average(
            ta, tb, A_u, t_start=t0, duration=tau
        )
    ).item())


def nonlinear_terms(q):
    th = np.tanh(beta * q)
    sech2 = 1.0 - th**2

    enl = mu * th**3
    psi = kappa * th**3
    chi = gamma * th**2

    denl = 3.0 * mu * beta * th**2 * sech2
    dpsi = 3.0 * kappa * beta * th**2 * sech2
    dchi = 2.0 * gamma * beta * th * sech2

    return enl, psi, chi, denl, dpsi, dchi


def hamiltonian(x):
    q, p, _ = x

    X = beta * q
    th = np.tanh(X)
    log_cosh = np.logaddexp(X, -X) - np.log(2.0)

    Hnl = (mu / beta) * (log_cosh - 0.5 * th**2)

    return 0.5 * (q**2 + p**2) + Hnl


def vector_field(x):
    q, p, z = x
    enl, psi, chi, _, _, _ = nonlinear_terms(q)

    a = q + enl

    return np.array([
        (omega + psi) * p - (nu + chi) * a,
        -(omega + psi) * a - (nu + chi) * p - c * z,
        c * p - rho * z,
    ])


def vector_field_jacobian(x):
    q, p, _ = x
    enl, psi, chi, denl, dpsi, dchi = nonlinear_terms(q)

    a = q + enl
    da = 1.0 + denl

    return np.array([
        [
            dpsi * p - dchi * a - (nu + chi) * da,
            omega + psi,
            0.0,
        ],
        [
            -dpsi * a - (omega + psi) * da - dchi * p,
            -(nu + chi),
            -c,
        ],
        [0.0, c, -rho],
    ])


def g(x):
    q, p, _ = x
    enl, psi, chi, _, _, _ = nonlinear_terms(q)

    a = q + enl

    return np.array([
        nu * enl - psi * p + chi * a,
        omega * enl + psi * a + chi * p,
        0.0,
    ])


def dissipation(x):
    q, p, z = x
    enl, _, chi, _, _, _ = nonlinear_terms(q)

    a = q + enl

    return (nu + chi) * (a**2 + p**2) + rho * z**2


def output(x):
    return x[1]


# ----------------------------------------------------------------------
# Independent DOP853 reference
# ----------------------------------------------------------------------

def reference_rhs(t, y):
    q, p = y
    enl, psi, chi, _, _, _ = nonlinear_terms(q)

    a = q + enl

    return np.array([
        (omega + psi) * p - (nu + chi) * a,
        -(omega + psi) * a
        - (nu + chi + c**2 / rho) * p
        + u(t),
    ])


print("Computing DOP853 reference...")

ref = solve_ivp(
    reference_rhs,
    (0.0, T),
    np.zeros(2),
    method="DOP853",
    rtol=1e-12,
    atol=1e-14,
    dense_output=True,
)

if not ref.success:
    raise RuntimeError(ref.message)


def reference(t):
    Y = ref.sol(t).T

    X = np.empty((len(t), 3))
    X[:, 0] = Y[:, 0]
    X[:, 1] = Y[:, 1]
    X[:, 2] = (c / rho) * Y[:, 1]

    return X


def errors(t, X):
    Xref = reference(t)

    H = np.array([hamiltonian(x) for x in X])
    Href = np.array([hamiltonian(x) for x in Xref])

    Ex = np.max(np.linalg.norm(X - Xref, axis=1)) / np.sqrt(3.0)
    EH = np.max(np.abs(H - Href))

    return float(Ex), float(EH)


# ----------------------------------------------------------------------
# BDF2-EOP-GSAV
# ----------------------------------------------------------------------

def relax(xbar, r_old, power, dt, order):
    Hshift = hamiltonian(xbar) + C0
    budget = r_old + dt * power

    if budget <= 0.0 or Hshift <= 0.0:
        raise RuntimeError("Nonpositive EOP budget or shifted Hamiltonian.")

    rtilde = budget / (
        1.0 + dt * dissipation(xbar) / Hshift
    )

    xi = rtilde / Hshift
    eta = 1.0 - (1.0 - xi) ** (order + 1)

    x = eta * xbar
    r = min(hamiltonian(x) + C0, budget)

    return x, r


def eop_bdf2(dt):
    steps = int(round(T / dt))
    dt = T / steps
    t = np.linspace(0.0, T, steps + 1)

    X = np.zeros((steps + 1, 3))
    X[0] = x0

    r = hamiltonian(x0) + C0

    M1 = E / dt + A
    M2 = 1.5 * E / dt + A

    lu1 = lu_factor(M1)
    lu2 = lu_factor(M2)

    # BDF1 start
    ub_prev = ubar(t[0], t[1])

    rhs = E @ X[0] / dt - g(X[0]) + B * ub_prev
    xbar = lu_solve(lu1, rhs)

    ybar_prev = output(xbar)
    X[1], r = relax(
        xbar, r, ub_prev * ybar_prev, dt, order=1
    )

    # BDF2
    for n in range(1, steps):
        ub_cur = ubar(t[n], t[n + 1])

        C2 = 1.5 * ub_cur - 0.5 * ub_prev
        xext = 2.0 * X[n] - X[n - 1]

        rhs = (
            (2.0 * E @ X[n] - 0.5 * E @ X[n - 1]) / dt
            - g(xext)
            + B * C2
        )

        xbar = lu_solve(lu2, rhs)
        ybar_cur = output(xbar)

        P2 = ub_cur * 0.5 * (ybar_cur + ybar_prev)

        X[n + 1], r = relax(
            xbar, r, P2, dt, order=2
        )

        ub_prev = ub_cur
        ybar_prev = ybar_cur

    return {
        "converged": True,
        "t": t,
        "x": X,
        "factorizations": 2,
        "linear_solves": steps,
    }


# ----------------------------------------------------------------------
# Implicit midpoint and Newton variants
# ----------------------------------------------------------------------

def midpoint_residual(z, xn, tmid, dt):
    xm = 0.5 * (z + xn)

    return (
        E @ (z - xn) / dt
        - vector_field(xm)
        - B * u(tmid)
    )


def midpoint_jacobian(z, xn, dt):
    xm = 0.5 * (z + xn)

    return E / dt - 0.5 * vector_field_jacobian(xm)


def newton_step(xn, tmid, dt, mode, frozen_lu=None):
    z = xn.copy()
    factorizations = 0
    linear_solves = 0

    lu = None

    if mode == "modified":
        lu = lu_factor(midpoint_jacobian(z, xn, dt))
        factorizations = 1

    elif mode == "simple":
        lu = frozen_lu

    for iteration in range(newton_maxiter + 1):
        F = midpoint_residual(z, xn, tmid, dt)
        fnorm = np.linalg.norm(F)

        if fnorm <= newton_tol:
            return {
                "x": z,
                "converged": True,
                "iterations": iteration,
                "factorizations": factorizations,
                "linear_solves": linear_solves,
            }

        if iteration == newton_maxiter:
            break

        if mode == "full":
            lu = lu_factor(midpoint_jacobian(z, xn, dt))
            factorizations += 1

        z += lu_solve(lu, -F)
        linear_solves += 1

    return {
        "x": z,
        "converged": False,
        "iterations": newton_maxiter,
        "factorizations": factorizations,
        "linear_solves": linear_solves,
    }


def implicit_midpoint(dt, mode):
    steps = int(round(T / dt))
    dt = T / steps
    t = np.linspace(0.0, T, steps + 1)

    X = np.zeros((steps + 1, 3))
    X[0] = x0

    iterations = []
    factorizations = 0
    linear_solves = 0

    frozen_lu = None

    if mode == "simple":
        M0 = E / dt - 0.5 * vector_field_jacobian(x0)
        frozen_lu = lu_factor(M0)
        factorizations = 1

    for n in range(steps):
        result = newton_step(
            X[n],
            0.5 * (t[n] + t[n + 1]),
            dt,
            mode,
            frozen_lu,
        )

        iterations.append(result["iterations"])
        factorizations += result["factorizations"]
        linear_solves += result["linear_solves"]

        if not result["converged"]:
            return {
                "converged": False,
                "fail_time": t[n + 1],
                "mean_newton": float(np.mean(iterations)),
                "max_newton": int(np.max(iterations)),
                "factorizations": factorizations,
                "linear_solves": linear_solves,
            }

        X[n + 1] = result["x"]

    return {
        "converged": True,
        "t": t,
        "x": X,
        "mean_newton": float(np.mean(iterations)),
        "max_newton": int(np.max(iterations)),
        "factorizations": factorizations,
        "linear_solves": linear_solves,
    }


# ----------------------------------------------------------------------
# Frozen-Jacobian diagnostic used in the paper
# ----------------------------------------------------------------------

def frozen_diagnostic(sol, dt):
    M0 = E / dt - 0.5 * vector_field_jacobian(x0)
    Df0 = vector_field_jacobian(x0)

    contraction = 0.0
    jacobian_variation = 0.0

    for n in range(sol["steps"] if "steps" in sol else len(sol["x"]) - 1):
        xm = 0.5 * (sol["x"][n] + sol["x"][n + 1])
        Dfm = vector_field_jacobian(xm)

        Mn = E / dt - 0.5 * Dfm
        G = np.eye(3) - np.linalg.solve(M0, Mn)

        contraction = max(
            contraction,
            float(np.linalg.norm(G, 2)),
        )

        jacobian_variation = max(
            jacobian_variation,
            float(np.linalg.norm(Dfm - Df0, 2)),
        )

    return contraction, jacobian_variation


# ----------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------

methods = {
    "BDF2-EOP": lambda dt: eop_bdf2(dt),
    "IMP-full": lambda dt: implicit_midpoint(dt, "full"),
    "IMP-modified": lambda dt: implicit_midpoint(dt, "modified"),
    "IMP-simple": lambda dt: implicit_midpoint(dt, "simple"),
}

rows = []
diagnostics = []

for dt in dts:
    print(f"\ndt={dt:g}")

    solutions = {}

    for name, solver in methods.items():
        times = []
        sol = None

        for rep in range(repeats):
            tic = time.perf_counter()
            candidate = solver(dt)
            elapsed = time.perf_counter() - tic

            sol = candidate

            if not candidate["converged"]:
                break

            times.append(elapsed)

        if not sol["converged"]:
            print(
                f"  {name:12s}: FAIL at t={sol['fail_time']:.3f}"
            )

            rows.append({
                "dt": dt,
                "method": name,
                "status": "fail",
                "runtime": np.nan,
                "E_x": np.nan,
                "E_H": np.nan,
                "mean_newton": sol.get("mean_newton", np.nan),
                "max_newton": sol.get("max_newton", np.nan),
                "factorizations": sol["factorizations"],
                "linear_solves": sol["linear_solves"],
            })

            continue

        runtime = float(np.median(times))
        Ex, EH = errors(sol["t"], sol["x"])

        solutions[name] = sol

        row = {
            "dt": dt,
            "method": name,
            "status": "conv",
            "runtime": runtime,
            "E_x": Ex,
            "E_H": EH,
            "mean_newton": sol.get("mean_newton", np.nan),
            "max_newton": sol.get("max_newton", np.nan),
            "factorizations": sol["factorizations"],
            "linear_solves": sol["linear_solves"],
        }

        rows.append(row)

        print(
            f"  {name:12s}: "
            f"time={runtime:.5f}s  "
            f"Ex={Ex:.3e}  EH={EH:.3e}  "
            f"fact={row['factorizations']}  "
            f"solves={row['linear_solves']}"
        )

    if "IMP-full" in solutions:
        chi, dDf = frozen_diagnostic(
            solutions["IMP-full"], dt
        )

        diagnostics.append({
            "dt": dt,
            "max_frozen_iteration_norm": chi,
            "max_jacobian_variation": dDf,
        })

        print(
            f"  frozen: ||I-M0^-1 Mn||={chi:.3f}, "
            f"||Df-Df0||={dDf:.3f}"
        )


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------

print("\nSolver comparison\n")
print(
    f"{'dt':>9} {'method':>13} {'status':>7} "
    f"{'time':>9} {'E_x':>11} {'E_H':>11} "
    f"{'mean N':>8} {'max N':>6} "
    f"{'fact':>6} {'solves':>7}"
)

for r in rows:
    print(
        f"{r['dt']:9.5f} "
        f"{r['method']:>13} "
        f"{r['status']:>7} "
        f"{r['runtime']:9.5f} "
        f"{r['E_x']:11.3e} "
        f"{r['E_H']:11.3e} "
        f"{r['mean_newton']:8.2f} "
        f"{r['max_newton']:6.0f} "
        f"{r['factorizations']:6.0f} "
        f"{r['linear_solves']:7.0f}"
    )


with open(OUT / "summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


with open(
    OUT / "frozen_jacobian.csv",
    "w",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=diagnostics[0].keys(),
    )
    writer.writeheader()
    writer.writerows(diagnostics)


print(f"\nResults written to {OUT}")