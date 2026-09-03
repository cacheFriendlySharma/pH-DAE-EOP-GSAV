import numpy as np
import scipy.linalg as sla
import scipy.sparse as sps
import scipy.sparse.linalg as spla


def factorize(A):
    if sps.issparse(A):
        return spla.splu(A.tocsc())

    return sla.lu_factor(np.asarray(A))


def solve_factorized(factorization,b):
    if isinstance(factorization,spla.SuperLU):
        return factorization.solve(b)

    return sla.lu_solve(factorization,b)