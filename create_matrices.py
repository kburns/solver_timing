"""
Create matrices for speed testing sparse solvers.

"""


import numpy as np
import pickle
import shelve
import parameters as params
import problem


# Build resolution array
log2_min = int(np.log2(params.min_res))
log2_max = int(np.log2(params.max_res))
resolutions = 2 ** np.arange(log2_min, log2_max+1)

# Build LHS matrices
LHS = [problem.build_LHS(Nz) for Nz in resolutions]

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
    file['resolutions'] = resolutions
    file['sizes'] = sizes
    file['LHS'] = LHS
    file['RHS'] = RHS

