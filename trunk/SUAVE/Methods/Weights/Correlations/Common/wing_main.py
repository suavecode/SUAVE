## @ingroup Methods-Weights-Correlations-Common 
# wing_main.py
#
# Created:  Jan 2014, A. Wendorff
# Modified: Feb 2014, A. Wendorff
#           Feb 2016, E. Botero
#           Jul 2017, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units
import numpy as np


# ----------------------------------------------------------------------
#   Wing Main
# ----------------------------------------------------------------------

## @ingroup Methods-Weights-Correlations-Common 
def wing_main(vehicle, wing):
    """ Calculate the wing weight of the aircraft based on the fully-stressed 
    bending weight of the wing box
    
    Assumptions:
        calculated total wing weight based on a bending index and actual data 
        from 15 transport aircraft 
    
    Source: 
        N/A
        
    Inputs:
        vehicle - data dictionary with vehicle properties                   [dimensionless]
        wing    - data dictionary with specific wing properties             [dimensionless]
    
    Outputs:
        weight - weight of the wing                  [kilograms]          
        
    Properties Used:
        N/A
    """

    S_gross_w = wing.areas.reference
    sweep_w = wing.sweeps.quarter_chord
    # unpack inputs
    span = wing.spans.projected / Units.ft  # Convert meters to ft
    sweep = sweep_w
    area = S_gross_w / Units.ft ** 2  # Convert meters squared to ft squared
    mtow = vehicle.mass_properties.max_takeoff / Units.lb  # Convert kg to lbs
    zfw = vehicle.mass_properties.max_zero_fuel / Units.lb  # Convert kg to lbs
    # Calculate weight of wing for traditional aircraft wing
    weight = 4.22 * area + 1.642 * 10. ** -6. * vehicle.envelope.ultimate_load * span ** 3. * (mtow * zfw) ** 0.5 \
             * (1. + 2. * wing.taper) / (wing.thickness_to_chord * (np.cos(sweep)) ** 2. * area * (1. + wing.taper))
    weight = weight * Units.lb  # Convert lb to kg

    return weight
