## @ingroup Methods-Weights-Correlations-Tube_Wing
# tail_horizontal.py
#
# Created:  Jan 2014, A. Wendorff
# Modified: Feb 2016, E. Botero  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units
import numpy as np

# ----------------------------------------------------------------------
#   Tail Horizontal
# ----------------------------------------------------------------------

## @ingroup Methods-Weights-Correlations-Tube_Wing
def tail_horizontal(b_h,sweep_h,Nult,S_h,TOW,mac_w,mac_h,l_w2h,t_c_h,exposed):      
    """ Calculate the weight of the horizontal tail in a standard configuration
    
    Assumptions:
        calculated weight of the horizontal tail including the elevator
        Assume that the elevator is 25% of the horizontal tail 
    
    Source: 
        Aircraft Design: A Conceptual Approach by Raymer
        
    Inputs:
        b_h - span of the horizontal tail                                                               [meters]
        sweep_h - sweep of the horizontal tail                                                          [radians]
        Nult - ultimate design load of the aircraft                                                     [dimensionless]
        S_h - area of the horizontal tail                                                               [meters**2]
        TOW - maximum takeoff weight of the aircraft                                                    [kilograms]
        mac_w - mean aerodynamic chord of the wing                                                      [meters]
        mac_h - mean aerodynamic chord of the horizontal tail                                           [meters]
        l_w2h - tail length (distance from the airplane c.g. to the horizontal tail aerodynamic center) [meters]
        t_c_h - thickness-to-chord ratio of the horizontal tail                                         [dimensionless]
        exposed - exposed area ratio for the horizontal tail                                            [dimensionless]
    
    Outputs:
        weight - weight of the horizontal tail                                                          [kilograms]
       
    Properties Used:
        N/A
    """   
    # unpack inputs
    span       = b_h / Units.ft # Convert meters to ft
    sweep      = sweep_h # Convert deg to radians
    area       = S_h / Units.ft**2 # Convert meters squared to ft squared
    mtow       = TOW / Units.lb # Convert kg to lbs
    l_w        = mac_w / Units.ft # Convert from meters to ft
    l_h        = mac_h / Units.ft # Convert from meters to ft
    length_w_h = l_w2h / Units.ft # Distance from mean aerodynamic center of wing to mean aerodynamic center of horizontal tail (Convert meters to ft)

    #Calculate weight of wing for traditional aircraft horizontal tail
    weight_English = (5.25*area+0.8*10.**(-6.)*Nult*span**3.*mtow*l_w*(exposed*area)**(1./2.)/(t_c_h*(np.cos(sweep)**2.)*length_w_h*area**1.5))

    weight = weight_English * Units.lbs # Convert from lbs to kg

    return weight