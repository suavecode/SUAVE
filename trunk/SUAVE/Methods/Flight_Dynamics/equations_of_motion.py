""" FlightDynamics.py: Methods for Flight Dynamics Analysis """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def equations_of_motion(segment):

    """  Equations of Motion """

    if segment.complex:
        dzdt = np.zeros((segment.N,segment.dofs*2)) + 0j
    else:
        dzdt = np.zeros((segment.N,segment.dofs*2))

    dzdt[:,0] = segment.vectors.V[:,0]                                  # dx/dt
    dzdt[:,1] = segment.vectors.Ftot[:,0]/segment.m                     # dVx/dt

    if segment.dofs == 2:
        dzdt[:,2] = segment.vectors.V[:,2]                              # dz/dt
        dzdt[:,3] = segment.vectors.Ftot[:,2]/segment.m - segment.g     # dVz/dt
    elif segment.dofs == 3:
        dzdt[:,2] = segment.vectors.V[:,1]                              # dy/dt
        dzdt[:,3] = segment.vectors.Ftot[:,1]/segment.m                 # dVy/dt
        dzdt[:,4] = segment.vectors.V[:,2]                              # dz/dt
        dzdt[:,5] = segment.vectors.Ftot[:,2]/segment.m - segment.g     # dVz/dt
    else:
        print "Error: degrees of freedom > 3 specified." 
        return []

    return dzdt