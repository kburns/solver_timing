"""Classes wrapping various linear solvers."""

import numpy as np
import logging
logger = logging.getLogger(__name__)

try:
    import scipy
    import scipy.linalg as sla
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except ImportError:
    logger.warning("Cannot import scipy")
    scipy = None

try:
    import sparseqr
except ImportError:
    logger.warning("Cannot import sparseqr")
    sparseqr = None

try:
    import pybanded
except ImportError:
    logger.warning("Cannot import pybanded")
    pybanded = None


solvers = []

def add_solver(library):
    """Filter solvers based on presence of required libraries."""
    def _add_solver(solver):
        if library:
            solvers.append(solver)
        return solver
    return _add_solver


class SparseSolver:
    sparse = True


class DenseSolver:
    sparse = False


@add_solver(scipy)
class UmfpackSpsolve(SparseSolver):
    """UMFPACK spsolve"""

    def __init__(self, matrix):
        self.matrix = matrix

    def solve(self, vector):
        return spla.spsolve(self.matrix, vector, use_umfpack=True)


@add_solver(scipy)
class SuperluNaturalSpsolve(SparseSolver):
    """SuperLU+NATURAL spsolve"""

    def __init__(self, matrix):
        self.matrix = matrix

    def solve(self, vector):
        return spla.spsolve(self.matrix, vector, permc_spec='NATURAL', use_umfpack=False)


@add_solver(scipy)
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


@add_solver(scipy)
class SuperluNaturalFactorized(SparseSolver):
    """SuperLU+NATURAL LU factorized solve"""

    def __init__(self, matrix):
        self.LU = spla.splu(matrix, permc_spec='NATURAL')

    def solve(self, vector):
        return self.LU.solve(vector)


@add_solver(scipy)
class SuperluColamdFactorized(SparseSolver):
    """SuperLU+COLAMD LU factorized solve"""

    def __init__(self, matrix):
        self.LU = spla.splu(matrix, permc_spec='COLAMD')

    def solve(self, vector):
        return self.LU.solve(vector)


@add_solver(scipy)
class ScipyBanded(DenseSolver):
    """Scipy banded solve"""

    def __init__(self, matrix):
        self.lu, self.ab = self.dia_to_banded(matrix)

    def solve(self, vector):
        return sla.solve_banded(self.lu, self.ab, vector, check_finite=False)

    @staticmethod
    def dia_to_banded(matrix):
        """Convert sparse DIA matrix to banded format."""
        matrix = sp.dia_matrix(matrix)
        u = max(0, max(matrix.offsets))
        l = max(0, max(-matrix.offsets))
        ab = np.zeros((u+l+1, matrix.shape[1]), dtype=matrix.dtype)
        ab[u-matrix.offsets] = matrix.data
        lu = (l, u)
        return lu, ab


@add_solver(sparseqr)
class SPQR_solve(SparseSolver):
    """SuiteSparse QR solve"""

    def __init__(self, matrix):
        self.matrix = matrix

    def solve(self, vector):
        return sparseqr.solve(self.matrix, vector)


@add_solver(pybanded)
class BandedQR(DenseSolver):
    """pybanded QR solve"""

    def __init__(self, matrix):
        matrix = pybanded.BandedMatrix.from_sparse(matrix)
        self.QR = pybanded.BandedQR(matrix)

    def solve(self, vector):
        return self.QR.solve(vector)

