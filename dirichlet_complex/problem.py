"""
Define problem for speed testing sparse solvers.

"""

import numpy as np
import dedalus.public as de


def build_LHS(Nz, bw, format):

    # Parameters
    dt = 1e-3
    kx = 50 * np.pi
    sigma = 1
    Prandtl = 1.
    Reynolds = 1e2
    ts = de.timesteppers.RK222

    # Create bases and domain
    z_basis = de.Chebyshev('z', Nz, interval=(-1, 1), dealias=3/2)
    domain = de.Domain([z_basis], grid_dtype=np.complex128)

    # 2D Boussinesq hydrodynamics
    problem = de.IVP(domain, variables=['p','b','u','w','bz','uz','wz'], ncc_cutoff=0, max_ncc_terms=bw)
    problem.meta[:]['z']['dirichlet'] = True
    problem.parameters['P'] = 1 / Reynolds / Prandtl
    problem.parameters['R'] = 1 / Reynolds
    problem.parameters['kx'] = kx
    problem.parameters['sigma'] = sigma
    problem.substitutions['Bz'] = "exp(-(z-1)**2 / 2 / sigma**2)"
    problem.substitutions['dx(A)'] = "1j*kx*A"
    problem.add_equation("dx(u) + wz = 0")
    problem.add_equation("dt(b) - P*(dx(dx(b)) + dz(bz)) + Bz*w      = 0")
    problem.add_equation("dt(u) - R*(dx(dx(u)) + dz(uz)) + dx(p)     = 0")
    problem.add_equation("dt(w) - R*(dx(dx(w)) + dz(wz)) + dz(p) - b = 0")
    problem.add_equation("bz - dz(b) = 0")
    problem.add_equation("uz - dz(u) = 0")
    problem.add_equation("wz - dz(w) = 0")
    problem.add_bc("left(b) = 0")
    problem.add_bc("left(u) = 0")
    problem.add_bc("left(w) = 0")
    problem.add_bc("right(b) = 0")
    problem.add_bc("right(u) = 0")
    if kx == 0:
        problem.add_bc("right(p) = 0")
    else:
        problem.add_bc("right(w) = 0",)

    # Build solver
    solver = problem.build_solver(ts)

    # Step solver to form pencil LHS
    for i in range(1):
        solver.step(dt)

    return solver.pencils[0].LHS
