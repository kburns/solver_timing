"""Speed-test sparse solvers."""

import numpy as np
import pickle
import shelve
import time
import solvers
import logging
import xarray
from memory_profiler import memory_usage
from collections import OrderedDict
logger = logging.getLogger(__name__)


def time_solver(solver, loops, matrix_file='matrices.db', timing_file='timing', timer=time.perf_counter):
    """Time solver over all saved matrices."""
    print("Timing solver: %s" %solver.__doc__)
    # Open data
    with shelve.open(matrix_file, 'r') as file:
        args = file['args']
        sizes = file['sizes']
        LHS = file['LHS']
        RHS = file['RHS']
    # Build coordinate arrays
    coords = OrderedDict()
    for key in args[0]:
        coords[key] = np.sort(list(set(arg[key] for arg in args)))
    shape = [len(c) for c in coords.values()]
    start_first = xarray.DataArray(data=np.zeros(shape), coords=coords.values(), dims=coords.keys())
    start_times = xarray.DataArray(data=np.zeros(shape), coords=coords.values(), dims=coords.keys())
    start_mem   = xarray.DataArray(data=np.zeros(shape), coords=coords.values(), dims=coords.keys())
    solve_first = xarray.DataArray(data=np.zeros(shape), coords=coords.values(), dims=coords.keys())
    solve_times = xarray.DataArray(data=np.zeros(shape), coords=coords.values(), dims=coords.keys())
    solve_mem   = xarray.DataArray(data=np.zeros(shape), coords=coords.values(), dims=coords.keys())
    # Loop over matrices
    for arg, lhs, rhs in zip(args, LHS, RHS):
        print("  %s" %arg)
        # Warm-up
        start = timer()
        s = solver(lhs)
        end = timer()
        start_first.loc[arg] = (end - start)
        # Compute startup timing
        start_mem = memory_usage(max_usage=True)
        start = timer()
        s = []
        for i in range(loops):
            s.append(solver(lhs))
        end = timer()
        end_mem = memory_usage(max_usage=True)
        start_times.loc[arg] = (end - start) / loops
        start_mem.loc[arg] = (end_mem - start_mem) / loops
        # Warm-up
        start = timer()
        x = s.solve(rhs)
        end = timer()
        solve_first.loc[arg] = (end - start)
        # Compute solve timing
        start_mem = memory_usage(max_usage=True)
        start = timer()
        x = []
        for i in range(loops):
            x.append(s.solve(lhs))
        end = timer()
        end_mem = memory_usage(max_usage=True)
        solve_times.loc[arg] = (end - start) / loops
        solve_mem.loc[arg] = (end_mem - start_mem) / loops
    ds = xarray.Dataset({'start_time': start_times,
                         'solve_time': solve_times})
    ds.attrs['loops'] = loops
    ds.attrs['solver_name'] = solver.__name__
    ds.attrs['solver_doc'] = solver.__doc__
    # Save dataset
    timing_file = timing_file + '_%s.pkl' %solver.__name__
    pickle.dump(ds, open(timing_file, 'wb'))


if __name__ == "__main__":
    import parameters as params
    # Time all solvers
    solvers = solvers.solvers
    for solver in solvers:
        time_solver(solver, params.loops)

