"""
Create matrices for speed testing sparse solvers.

"""


import numpy as np
import pickle
import shelve
import parameters as params
import problem
from itertools import product


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
with shelve.open('matrices.dat', 'n', protocol=pickle.HIGHEST_PROTOCOL) as file:
    file['args'] = prod_args
    file['sizes'] = sizes
    file['LHS'] = LHS
    file['RHS'] = RHS

