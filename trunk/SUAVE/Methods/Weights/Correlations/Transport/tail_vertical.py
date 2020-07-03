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
def tail_vertical(S_v,Nult,b_v,TOW,t_c_v,sweep_v,S_gross_w,t_tail,rudder_fraction = 0.25):      
    """ Calculate the weight of the vertical fin of an aircraft without the weight of 
    the rudder and then calculate the weight of the rudder 
    
    Assumptions:
        Vertical tail weight is the weight of the vertical fin without the rudder weight.
        Rudder occupies 25% of the S_v and weighs 60% more per unit area.     
        
    Source: 
        N/A 
        
    Inputs:
        S_v - area of the vertical tail (combined fin and rudder)                      [meters**2]
        Nult - ultimate load of the aircraft                                           [dimensionless]
        b_v - span of the vertical                                                     [meters]
        TOW - maximum takeoff weight of the aircraft                                   [kilograms]
        t_c_v - thickness-to-chord ratio of the vertical tail                          [dimensionless]
        sweep_v - sweep angle of the vertical tail                                     [radians]
        S_gross_w - wing gross area                                                    [meters**2]
        t_tail - factor to determine if aircraft has a t-tail                          [dimensionless]
        rudder_fraction - fraction of the vertical tail that is the rudder             [dimensionless]
    
    Outputs:
        output - a dictionary with outputs:
            wt_tail_vertical - weight of the vertical fin portion of the vertical tail [kilograms]
            wt_rudder - weight of the rudder on the aircraft                           [kilograms]
  
    Properties Used:
        N/A
    """      
    # unpack inputs
    span  = b_v / Units.ft # Convert meters to ft
    sweep = sweep_v # Convert deg to radians
    area  = S_v / Units.ft**2 # Convert meters squared to ft squared
    mtow  = TOW / Units.lb # Convert kg to lbs
    Sref  = S_gross_w / Units.ft**2 # Convert from meters squared to ft squared  
    
    # Determine weight of the vertical portion of the tail
    if t_tail == "yes": 
        T_tail_factor = 1.25 # Weight of vertical portion of the T-tail is 25% more than a conventional tail
    else: 
        T_tail_factor = 1.0 
    
    # Calculate weight of wing for traditional aircraft vertical tail without rudder
    tail_vert_English = T_tail_factor * (2.62*area+1.5*10.**(-5.)*Nult*span**3.*(8.+0.44*mtow/Sref)/(t_c_v*(np.cos(sweep)**2.))) 
    
    # packup outputs    
    
    output = Data()
    output.wt_tail_vertical = tail_vert_English * Units.lbs # Convert from lbs to kg
    output.wt_rudder        = output.wt_tail_vertical * rudder_fraction * 1.6

    return output
