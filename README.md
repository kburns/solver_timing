# Dedalus Solver Timings

This is a repository for testing the speed of different python-wrapped solvers for typical matrices produced in Dedalus IVPs.

## Adding solvers

To add a solver, simply create a wrapper following the examples in `solvers.py`.

## Running tests

To run, make a subdirectory with the following files, following the examples:

- problem.py
- parameters.py
- dedalus.cfg (optional)

Then execute the following top-level scripts in order from the subdirectory:

1. create_matrices.py
2. run_timings.py
3. plot_timings.py

## Contact

Maintained by [Keaton Burns](http://keaton-burns.com) at <https://bitbucket.org/kburns/solver_timing>.