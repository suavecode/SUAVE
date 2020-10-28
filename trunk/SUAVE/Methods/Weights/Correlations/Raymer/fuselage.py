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

## @ingroup Methods-Weights-Correlations-Raymer
def fuselage_weight_Raymer(vehicle, fuse, settings):
    """ Calculate the weight of the fuselage of a transport aircraft based on the Raymer method

        Assumptions:
            No fuselage mounted landing gear
            1 cargo door

        Source:
            Aircraft Design: A Conceptual Approach (2nd edition)

        Inputs:
            vehicle - data dictionary with vehicle properties                   [dimensionless]
                -.mass_properties.max_takeoff: MTOW                             [kg]
                -.envelope.ultimate_load: ultimate load factor (default: 3.75)
                -.wings['main_wing']: data dictionary with main wing properties
                    -.taper: wing taper ratio
                    -.sweeps.quarter_chord: quarter chord sweep                 [rad]
            fuse - data dictionary with specific fuselage properties            [dimensionless]
                -.lenghts.total: total length                                   [m]
                -.width: fuselage width                                         [m]
                -.heights.maximum: maximum height of the fuselage               [m]

        Outputs:
            weight_fuse - weight of the fuselage                                [kilograms]

        Properties Used:
            N/A
    """
    Klg     = settings.Raymer.fuselage_mounted_landing_gear_factor
    DG      = vehicle.mass_properties.max_takeoff / Units.lbs
    L       = fuse.lengths.total / Units.ft
    fuse_w  = fuse.width / Units.ft
    fuse_h  = fuse.heights.maximum / Units.ft
    
    Kdoor   = 1.06  # Assuming 1 cargo door
    D       = (fuse_w + fuse_h) / 2.
    Sf      = np.pi * (L / D - 1.7) * D ** 2  # Fuselage wetted area, ft**2
    wing    = vehicle.wings['main_wing']
    Kws     = 0.75 * (1 + 2 * wing.taper) / (1 + wing.taper) * (wing.spans.projected / Units.ft *
                                                            np.tan(wing.sweeps.quarter_chord)) / L

    weight_fuse = 0.328 * Kdoor * Klg * (DG * vehicle.envelope.ultimate_load) ** 0.5 * L ** 0.25 * \
                 Sf ** 0.302 * (1 + Kws) ** 0.04 * (L / D) ** 0.1
    return weight_fuse * Units.lbs
