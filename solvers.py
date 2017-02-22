"""Harness for testing matrix solver speed."""

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla


solvers = []
def add_solver(solver):
    solvers.append(solver)
    return solver


class SparseSolver:
    sparse = True


class DenseSolver:
    sparse = False


@add_solver
class UmfpackSpsolve(SparseSolver):
    """UMFPACK spsolve"""

    def __init__(self, matrix):
        self.matrix = matrix

    def solve(self, vector):
        return spla.spsolve(self.matrix, vector, use_umfpack=True)


@add_solver
class SuperluNaturalSpsolve(SparseSolver):
    """SuperLU+NATURAL spsolve"""

    def __init__(self, matrix):
        self.matrix = matrix

    def solve(self, vector):
        return spla.spsolve(self.matrix, vector, permc_spec='NATURAL', use_umfpack=False)


@add_solver
class SuperluColamdSpsolve(SparseSolver):
    """SuperLU+COLAMD spsolve"""

    def __init__(self, matrix):
        self.matrix = matrix

    def solve(self, vector):
        return spla.spsolve(self.matrix, vector, permc_spec='COLAMD', use_umfpack=False)


@add_solver
class UmfpackFactorized(SparseSolver):
    """UMFPACK LU factorized solve"""

    def __init__(self, matrix):
        self.LU = spla.factorized(matrix)

    def solve(self, vector):
        return self.LU(vector)


@add_solver
class SuperluNaturalFactorized(SparseSolver):
    """SuperLU+NATURAL LU factorized solve"""

    def __init__(self, matrix):
        self.LU = spla.splu(matrix, permc_spec='NATURAL')

    def solve(self, vector):
        return self.LU.solve(vector)


@add_solver
class SuperluColamdFactorized(SparseSolver):
    """SuperLU+COLAMD LU factorized solve"""

    def __init__(self, matrix):
        self.LU = spla.splu(matrix, permc_spec='COLAMD')

    def solve(self, vector):
        return self.LU.solve(vector)


def dia_to_banded(matrix):
    """Convert sparse DIA matrix to banded format."""
    matrix = sp.dia_matrix(matrix)
    u = max(0, max(matrix.offsets))
    l = max(0, max(-matrix.offsets))
    ab = np.zeros((u+l+1, matrix.shape[1]), dtype=matrix.dtype)
    ab[u-matrix.offsets] = matrix.data
    lu = (l, u)
    return lu, ab


@add_solver
class ScipyBanded(DenseSolver):
    """Scipy banded solve"""

    def __init__(self, matrix):
        self.lu, self.ab = dia_to_banded(matrix)

    def solve(self, vector):
        return sla.solve_banded(self.lu, self.ab, vector, check_finite=False)

