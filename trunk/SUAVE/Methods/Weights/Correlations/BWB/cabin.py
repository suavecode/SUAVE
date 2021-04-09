## @ingroup Methods-Weights-Correlations-BWB
# cabin.py
# 
# Created:  Jun 2014, T. Momose
# Modified: Feb 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Cabin
# ----------------------------------------------------------------------

## @ingroup Methods-Weights-Correlations-BWB
def cabin(cabin_area, TOGW):
    """ Weight estimate for the cabin (forward section of centerbody) of a BWB.
    Regression from FEA by K. Bradley (George Washington University).
    
    Assumptions:
        -The centerbody uses a pressurized sandwich composite structure
        -Ultimate cabin pressure differential of 18.6psi
        -Critical flight condition: +2.5g maneuver at maximum TOGW
    
    Sources:
        Bradley, K. R., "A Sizing Methodology for the Conceptual Design of 
        Blended-Wing-Body Transports," NASA/CR-2004-213016, 2004.
    
    Inputs:
        cabin_area - the planform area representing the passenger cabin  [meters**2]
        TOGW - Takeoff gross weight of the aircraft                      [kilograms]
    Outputs:
        cabin_wt - the estimated structural weight of the BWB cabin      [kilograms]
            
    Properties Used:
    N/A
    """       
    
    # convert to imperial units
    S_cab    = cabin_area / Units.feet ** 2.0
    W        = TOGW       / Units.pounds
    
    cabin_wt = 5.698865 * 0.316422 * (W ** 0.166552) * S_cab ** 1.061158
    
    # convert to base units
    cabin_wt = cabin_wt * Units.pounds
    
    return cabin_wt