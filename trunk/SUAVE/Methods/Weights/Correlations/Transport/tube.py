## @ingroup Methods-Weights-Correlations-Tube_Wing
# tube.py
#
# Created:  Jan 2014, A. Wendorff
# Modified: Feb 2014, A. Wendorff
#           Feb 2016, E. Botero  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units
import numpy as np

# ----------------------------------------------------------------------
#   Tube
# ----------------------------------------------------------------------

## @ingroup Methods-Weights-Correlations-Tube_Wing
def tube(S_fus, diff_p_fus, w_fus, h_fus, l_fus, Nlim, wt_zf, wt_wing, wt_propulsion, wing_c_r):
    """ Calculate the weight of a fuselage in the state tube and wing configuration
    
    Assumptions:
        fuselage in a standard wing and tube configuration         
    
    Source: 
        N/A 
        
    Inputs:
        S_fus - fuselage wetted area                                           [meters**2]
        diff_p_fus - Maximum fuselage pressure differential                    [Pascal]
        w_fus - width of the fuselage                                          [meters]
        h_fus - height of the fuselage                                         [meters]
        l_fus - length of the fuselage                                         [meters]
        Nlim - limit load factor at zero fuel weight of the aircraft           [dimensionless]
        wt_zf - zero fuel weight of the aircraft                               [kilograms]
        wt_wing - weight of the wing of the aircraft                           [kilograms]
        wt_propulsion - weight of the entire propulsion system of the aircraft [kilograms]
        wing_c_r - wing root chord                                             [meters]
        
    Outputs:
        weight - weight of the fuselage                                        [kilograms]
            
    Properties Used:
        N/A
    """     
    # unpack inputs
    
    diff_p = diff_p_fus / (Units.force_pound / Units.ft**2) # Convert Pascals to lbs/ square ft
    width = w_fus / Units.ft # Convert meters to ft
    height = h_fus / Units.ft # Convert meters to ft
   
    # setup
    length = l_fus - wing_c_r/2. 
    length = length / Units.ft # Convert meters to ft
    weight = (wt_zf - wt_wing - wt_propulsion) / Units.lb # Convert kg to lbs
    area = S_fus / Units.ft**2 # Convert square meters to square ft 
    
    #process
    
    # Calculate fuselage indices
    I_p = 1.5 *10**-3. * diff_p * width
    I_b = 1.91 *10 **-4. * Nlim * weight * length / height**2.
   
    
    if I_p > I_b : I_f = I_p
    else : I_f = (I_p**2. + I_b**2.)/(2.*I_b)
        
    #Calculate weight of wing for traditional aircraft vertical tail without rudder
    fuselage_weight = ((1.051+0.102*I_f) * area)  * Units.lb # Convert from lbs to kg
    
    return fuselage_weight