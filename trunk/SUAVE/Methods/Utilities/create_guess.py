""" Utilities.py: Mathematical tools and numerical integration methods for ODEs """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Structure import Data
from SUAVE.Attributes.Results import Segment

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def create_guess(problem,options):

    # d/dt, integration operators 
    D = problem.D/problem.tf; I = problem.I*problem.tf
    D[0,:] = 0.0; D[0,0] = 1.0

    zs = np.zeros((problem.Npoints,problem.Nstate))
    for j in range(problem.Nstate):
        zs[:,j] = problem.z0[j]*np.ones(problem.Npoints)
    
    if problem.Ncontrol > 0:
        zc = np.zeros((problem.Npoints,problem.Ncontrol))
        for j in range(problem.Ncontrol):
            zc[:,j] = problem.c0[j]*np.ones(problem.Npoints)

    dz = np.ones(problem.Nstate)
    zs_new = np.zeros_like(zs)

    # FPI with fixed final time
    while max(dz) > options.tol_solution:        

        if problem.Ncontrol > 0:
            rhs = problem.f(np.append(zs,zc,axis=1))
            rhs[0,range(problem.Nstate)] = problem.z0
        else:
            rhs = problem.dynamics(zs,D,I)
            rhs[0,:] = problem.z0

        for j in range(problem.Nstate):

            zs_new[:,j] = np.linalg.solve(D,rhs[:,j])
            dz[j] = np.linalg.norm(zs_new[:,j] - zs[:,j])
        
        zs = zs_new.copy()
    
    # flatten z array
    z = np.reshape(zs[1:,:],((options.Npoints-1)*problem.Nstate,1),order="F")

    # append control vars, if necessary
    if problem.Ncontrol > 0:
        z = np.append(z,np.reshape(zc,(problem.Npoints*problem.Ncontrol,1),order="F"))

    return z