"""
Define problem for speed testing sparse solvers.

"""

import numpy as np
import dedalus.public as de


def build_LHS(Nz):

    # Parameters
    dt = 1e-3
    kx = 50 * np.pi
    sigma = 1
    Prandtl = 1.
    Reynolds = 1e2
    ts = de.timesteppers.SBDF3

    # Create bases and domain
    z_basis = de.Fourier('z', Nz, interval=(-1, 1), dealias=3/2)
    domain = de.Domain([z_basis], grid_dtype=np.complex128)

    # 2D Boussinesq hydrodynamics
    problem = de.IVP(domain, variables=['p','b','u','w'])
    problem.parameters['P'] = 1 / Reynolds / Prandtl
    problem.parameters['R'] = 1 / Reynolds
    problem.parameters['kx'] = kx
    problem.parameters['sigma'] = sigma
    problem.substitutions['Bz'] = "1"
    problem.substitutions['dx(A)'] = "1j*kx*A"
    problem.add_equation("dt(b) - P*(dx(dx(b)) + dz(dz(b))) + Bz*w      = 0")
    problem.add_equation("dt(u) - R*(dx(dx(u)) + dz(dz(u))) + dx(p)     = 0")
    problem.add_equation("dt(w) - R*(dx(dx(w)) + dz(dz(w))) + dz(p) - b = 0")
    if kx == 0:
        problem.add_equation("dx(u) + dz(w) = 0", condition="(nz != 0)")
        problem.add_equation("p = 0", condition="(nz == 0)")
    else:
        problem.add_equation("dx(u) + dz(w) = 0")

    # Build solver
    solver = problem.build_solver(ts)

    # Step solver to form pencil LHS
    for i in range(10):
        solver.step(dt)

    return solver.pencils[0].LHS
