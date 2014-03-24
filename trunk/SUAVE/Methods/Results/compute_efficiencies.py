""" Results.py: Methods post-processing results """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def compute_efficiencies(results,summary=False):

    # evaluate each segment 
    for i in range(len(results.Segments)):

        segment = results.Segments[i]
        segment.P_fuel, segment.P_e = segment.config.Propulsors.power_flow(segment.eta,segment)

        # propulsive efficiency (mass flow)
        mask = (segment.P_fuel > 0.0)
        segment.efficiency.propellant = np.zeros_like(segment.F)
        segment.efficiency.propellant[mask] = segment.F[mask]*segment.V[mask]/segment.P_fuel[mask]

        # propulsive efficiency (electric) 
        mask = (segment.P_e > 0.0)
        segment.efficiency.electric = np.zeros_like(segment.F)
        segment.efficiency.electric[mask] = segment.F[mask]*segment.V[mask]/segment.P_e[mask]

    return