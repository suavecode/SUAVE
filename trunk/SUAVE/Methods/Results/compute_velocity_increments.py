""" Results.py: Methods post-processing results """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def compute_velocity_increments(results,summary=False):

    # evaluate each segment 
    for i in range(len(results.Segments)):

        segment = results.Segments[i]
        segment.P_fuel, segment.P_e = segment.config.Propulsors.power_flow(segment.eta,segment)

        # time integration operator
        I = segment.integrate

        # DV to gravity 
        segment.DV.gravity = np.dot(I,segment.g*segment.vectors.V[:,2])         # m/s

        # DV to drag
        segment.DV.drag = np.dot(I,segment.D*segment.V/segment.m)               # m/s

        # DV to drag
        FV = np.sum(segment.vectors.F*segment.vectors.V,axis=1)
        segment.DV.thrust = np.dot(I,FV/segment.m)                              # m/s

    return