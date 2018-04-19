## @ingroup Methods-Weights-Correlations-General_Aviation
# tail_vertical.py
#
# Created:  Feb 2018, M. Vegh
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units, Data
import numpy as np

# ----------------------------------------------------------------------
#   Tail Vertical
# ----------------------------------------------------------------------
## @ingroup Methods-Weights-Correlations-General_Aviation
def tail_vertical(S_v, AR_v, sweep_v, q_c, taper_v, t_c_v, Nult,TOW,t_tail, rudder_fraction = 0.25):      
    """
        Calculate the weight of the vertical fin of an aircraft without the weight of the rudder and then calculate the weight of the rudder        
        
        Source:
            Raymer,  Aircraft Design: A Conceptual Approach (pg 460 in 4th edition)
        
        Inputs:
            S_v - area of the vertical tail (combined fin and rudder)          [meters**2]
            M_w -mass of wing                                                  [kilograms]
            AR_v -aspect ratio of vertial tail                                 [dimensionless]
            sweep_v - sweep angle of the vertical tail                         [radians]
            q_c - dynamic pressure at cruise                                   [Pascals]
            taper_v - taper ratio of vertical tail                             [dimensionless]
            t_c_v -thickness to chord ratio of wing                            [dimensionless]
            Nult - ultimate load of the aircraft                               [dimensionless]
            TOW - maximum takeoff weight of the aircraft                       [kilograms]
            S_gross_w - wing gross area                                        [meters**2]
            t_tail - flag to determine if aircraft has a t-tail                [dimensionless]
            rudder_fraction - fraction of the vertical tail that is the rudder [dimensionless]
        
        Outputs:
            output - a dictionary with outputs:
                wt_tail_vertical - weight of the vertical fin portion of the vertical tail [kilograms]
            
        Assumptions:
            Vertical tail weight is the weight of the vertical fin without the rudder weight.
           
   """     
    # unpack inputs
    W_0   = TOW / Units.lb # Convert kg to lbs
    S_vt  = S_v/ Units.ft**2 # Convert from meters squared to ft squared  
    q     = q_c /(Units.force_pound / Units.ft**2.)

    # Determine weight of the vertical portion of the tail
    if t_tail == "yes": 
        T_tail_factor = 1.# Weight of vertical portion of the T-tail is 25% more than a conventional tail
    else: 
        T_tail_factor = 0.

    # Calculate weight of wing for traditional aircraft vertical tail without rudder
    tail_vert_English = .073*(1+.2*T_tail_factor)*((Nult*W_0)**(.376))*(q**.122)*(S_vt**.873)*((100.*t_c_v/np.cos(sweep_v))**(-.49))*((AR_v/(np.cos(sweep_v)**2.))**.357)*(taper_v**.039)

    # packup outputs    
    output                  = Data()
    output.wt_tail_vertical = tail_vert_English * Units.lbs # Convert from lbs to kg

    return output
