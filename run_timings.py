"""
Script for timing solver performance.
"""


import numpy as np
import pickle
import shelve
import time
import solvers
import parameters as params


# Setup timer and solvers
timer = time.perf_counter
loops = params.loops
if params.sparse_only:
    solvers = [s for s in solvers.solvers if s.sparse]
else:
    solvers = solvers.solvers

# Open data
with shelve.open('matrices.dat', 'r') as file:
    Args = file['args']
    sizes = file['sizes']
    LHS = file['LHS']
    RHS = file['RHS']

# Create reference solutions
solver = solvers[0]
X_ref = []
for A, b, in zip(LHS, RHS):
    s = solver(A)
    X_ref.append(s.solve(b))
all_match = True

# Time solvers
start_times = []
solve_times = []

# Loop over solvers
for solver in solvers:

    print(solver.__doc__)
    start_time = []
    solve_time = []

    # Loop over resolutions
    for args, A, b, x_ref in zip(Args, LHS, RHS, X_ref):
        print(str(args), end=': ')
        # Warm-up
        s = solver(A)
        # Compute startup timing
        start = timer()
        for i in range(loops):
            s = solver(A)
        end = timer()
        start_time.append((end-start)/loops)
        # Warm-up and check solver consistency
        x = s.solve(b)
        match = np.allclose(x, x_ref)
        print('match:', match)
        all_match = all_match and match
        # Compute solve timing
        start = timer()
        for i in range(loops):
            s.solve(b)
        end = timer()
        solve_time.append((end-start)/loops)

    start_times.append(np.array(start_time))
    solve_times.append(np.array(solve_time))

# Save data
solver_names = [s.__name__ for s in solvers]
solver_docs = [s.__doc__ for s in solvers]
with shelve.open('timings.dat', 'n', protocol=pickle.HIGHEST_PROTOCOL) as file:
    file['loops'] = loops
    file['args'] = Args
    file['sizes'] = sizes
    file['solver_names'] = solver_names
    file['solver_docs'] = solver_docs
    file['start_times'] = start_times
    file['solve_times'] = solve_times

print('All match:', all_match)

