"""
Define problem for speed testing sparse solvers.

"""

import numpy as np
import dedalus.public as de


def build_LHS(Nz, bw, format, entry_cutoff=0):

    # Parameters
    dt = 1e-3
    kx = 50 * np.pi
    sigma = 1
    ts = de.timesteppers.RK222

    # Create bases and domain
    z1 = de.Chebyshev('z', Nz, interval=(-1, 1), dealias=3/2)
    z2 = de.Chebyshev('z', Nz, interval=(1, 2), dealias=3/2)
    z3 = de.Chebyshev('z', Nz, interval=(2, 3), dealias=3/2)
    z_basis = de.Compound('z', [z1,z2,z3])
    domain = de.Domain([z_basis], grid_dtype=np.complex128)

    # 2D Boussinesq hydrodynamics
    problem = de.IVP(domain, variables=['T','Tz'], ncc_cutoff=0, max_ncc_terms=bw, entry_cutoff=entry_cutoff)
    problem.meta[:]['z']['dirichlet'] = True
    problem.parameters['kx'] = kx
    problem.parameters['sigma'] = sigma
    problem.substitutions['kappa'] = "exp(-(z-1)**2 / 2 / sigma**2)"
    problem.substitutions['dx(A)'] = "1j*kx*A"
    problem.add_equation("dt(T) - dx(kappa*dx(T)) - dz(kappa*Tz) = 0")
    problem.add_equation("Tz - dz(T) = 0")
    problem.add_bc("left(T) = 0")
    problem.add_bc("right(T) = 0")

    # Build solver
    solver = problem.build_solver(ts)

    # Step solver to form pencil LHS
    for i in range(1):
        solver.step(dt)

    return solver, solver.pencils[0].LHS.asformat(format)

