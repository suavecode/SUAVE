""" Utilities.py: Mathematical tools and numerical integration methods for ODEs """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
#import ad
from scipy.optimize import root   #, fsolve, newton_krylov
from SUAVE.Structure import Data
from SUAVE.Attributes.Results import Segment

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def create_state_data(z,u,problem,eta=[]):

    N, m = np.shape(z)

    # scale data
    if problem.dofs == 2:                      # 2-DOF
        z[:,0] *= problem.scale.L
        z[:,1] *= problem.scale.V
        z[:,2] *= problem.scale.L
        z[:,3] *= problem.scale.V
    elif problem.dofs == 3:                    # 3-DOF
        z[:,0] *= problem.scale.L
        z[:,1] *= problem.scale.V
        z[:,2] *= problem.scale.L
        z[:,3] *= problem.scale.V
        z[:,4] *= problem.scale.L
        z[:,5] *= problem.scale.V
    else:
        print "something went wrong in dimensionalize"
        return []
    
    state = State()
    if problem.powered:
        state.compute_state(z,u,problem.planet,problem.atmosphere, \
            problem.config.Functions.Aero,problem.config.Functions.Propulsion,eta,problem.flags)
    else:
        state.compute_state(z,u,problem.planet,problem.atmosphere, \
            problem.config.Functions.Aero,flags=problem.flags)

    return state