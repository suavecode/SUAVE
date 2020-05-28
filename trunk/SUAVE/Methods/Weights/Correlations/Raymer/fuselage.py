## @ingroup Methods-Weights-Correlations-Raymer
# fuselage.py
#
# Created:  May 2020, W. Van Gijseghem
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Units
import numpy as np

def fuselage_weight_Raymer(vehicle, fuse):
    """ Calculate the weight of the fuselage of a transport aircraft based on the Raymer method

        Assumptions:
            No fuselage mounted landing gear
            1 cargo door

        Source:
            Aircraft Design: A Conceptual Approach

        Inputs:
            vehicle - data dictionary with vehicle properties                   [dimensionless]
            fuse - data dictionary with specific fuselage properties            [dimensionless]

        Outputs:
            weight_fuse - weight of the fuselage                                [kilograms]

        Properties Used:
            N/A
    """
    Kdoor = 1.06  # Assuming 1 cargo door
    Klg = 1  # No fuselage mounted landing gear
    DG = vehicle.mass_properties.max_takeoff / Units.lbs
    L = fuse.lengths.total / Units.ft
    D = (fuse.width +
         fuse.heights.maximum) / 2. * 1 / Units.ft
    Sf = np.pi * (L / D - 1.7) * D ** 2  # Fuselage wetted area, ft**2
    wing = vehicle.wings['main_wing']
    Kws = 0.75 * (1 + 2 * wing.taper) / (1 + wing.taper) * (wing.spans.projected / Units.ft *
                                                            np.tan(wing.sweeps.quarter_chord)) / L

    weight_fuse = 0.328 * Kdoor * Klg * (DG * vehicle.envelope.ultimate_load) ** 0.5 * L ** 0.25 * \
                 Sf ** 0.302 * (1 + Kws) ** 0.04 * (L / D) ** 0.1
    return weight_fuse * Units.lbs
