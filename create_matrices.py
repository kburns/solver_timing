"""Create matrices for speed testing sparse solvers."""

import numpy as np
import pickle
from itertools import product


def create_matrices(problem, params, matrix_file='matrices.pkl'):
    # Build argument iterator
    keys = params.args.keys()
    values = params.args.values()
    prod_args = [dict(zip(keys, valset)) for valset in product(*values)]
    # Build LHS matrices
    LHS = [problem.build_LHS(**args) for args in prod_args]
    # Build size array
    sizes = np.array([A.shape[0] for A in LHS])
    # Build RHS vectors
    RHS = []
    for A, size in zip(LHS, sizes):
        if A.dtype == np.float64:
            b = np.random.randn(size)
        elif A.dtype == np.complex128:
            b = np.random.randn(size) + 1j*np.random.randn(size)
        RHS.append(b)
    # Save data
    data = {'args': prod_args,
            'sizes': sizes,
            'LHS': LHS,
            'RHS': RHS}
    pickle.dump(data, open(matrix_file, 'wb'), protocol=-1)

if __name__ == "__main__":
    import parameters as params
    import problem
    create_matrices(problem, params)

