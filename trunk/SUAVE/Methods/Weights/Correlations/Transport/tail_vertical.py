## @ingroup Methods-Weights-Correlations-Tube_Wing
# tail_vertical.py
#
# Created:  Jan 2014, A. Wendorff
# Modified: Feb 2016, E. Botero  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units, Data
import numpy as np


# ----------------------------------------------------------------------
#   Tail Vertical
# ----------------------------------------------------------------------

## @ingroup Methods-Weights-Correlations-Tube_Wing
def tail_vertical(vehicle, wing, rudder_fraction=0.25):
    """ Calculate the weight of the vertical fin of an aircraft without the weight of 
    the rudder and then calculate the weight of the rudder 
    
    Assumptions:
        Vertical tail weight is the weight of the vertical fin without the rudder weight.
        Rudder occupies 25% of the S_v and weighs 60% more per unit area.     
        
    Source: 
        N/A 
        
    Inputs:
        S_v - area of the vertical tail (combined fin and rudder)                      [meters**2]
        vehicle.envelope.ultimate_load - ultimate load of the aircraft                 [dimensionless]
        wing.spans.projected - span of the vertical                                    [meters]
        vehicle.mass_properties.max_takeoff - maximum takeoff weight of the aircraft   [kilograms]
        wing.thickness_to_chord- thickness-to-chord ratio of the vertical tail         [dimensionless]
        wing.sweeps.quarter_chord - sweep angle of the vertical tail                   [radians]
        vehicle.reference_area - wing gross area                                       [meters**2]
        wing.t_tail - factor to determine if aircraft has a t-tail                     [dimensionless]
        rudder_fraction - fraction of the vertical tail that is the rudder             [dimensionless]
    
    Outputs:
        output - a dictionary with outputs:
            wt_tail_vertical - weight of the vertical fin portion of the vertical tail [kilograms]
            wt_rudder - weight of the rudder on the aircraft                           [kilograms]
  
    Properties Used:
        N/A
    """
    # unpack inputs
    span    = wing.spans.projected / Units.ft  # Convert meters to ft
    sweep   = wing.sweeps.quarter_chord  # Convert deg to radians
    area    = wing.areas.reference / Units.ft ** 2  # Convert meters squared to ft squared
    mtow    = vehicle.mass_properties.max_takeoff / Units.lb  # Convert kg to lbs
    Sref    = vehicle.reference_area / Units.ft ** 2  # Convert from meters squared to ft squared
    t_c_v   = wing.thickness_to_chord
    # Determine weight of the vertical portion of the tail
    if wing.t_tail == "yes":
        T_tail_factor = 1.25  # Weight of vertical portion of the T-tail is 25% more than a conventional tail
    else:
        T_tail_factor = 1.0

        # Calculate weight of wing for traditional aircraft vertical tail without rudder
    tail_vert_English = T_tail_factor * (
                2.62 * area + 1.5 * 10. ** (-5.) * vehicle.envelope.ultimate_load * span ** 3. * (8. + 0.44 * mtow / Sref) / (
                    t_c_v * (np.cos(sweep) ** 2.)))

    tail_weight = tail_vert_English * Units.lbs
    tail_weight += tail_weight * rudder_fraction * 1.6

    return tail_weight
