# tail_vertical.py
#
# Created:  Mar 2016, M. Vegh
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units, Data
import numpy as np

# ----------------------------------------------------------------------
#   Tail Vertical
# ----------------------------------------------------------------------

def tail_vertical(S_v, AR_v, sweep_v, q_c, taper_v, t_c_v, Nult,TOW,t_tail, rudder_fraction = 0.25):      
    """ output = SUAVE.Methods.Weights.Correlations.Tube_Wing.tail_vertical(S_v,Nult,b_v,TOW,t_c_v,sweep_v,S_gross_w,t_tail)
        Calculate the weight of the vertical fin of an aircraft without the weight of the rudder and then calculate the weight of the rudder        
        
        Inputs:
            S_v - area of the vertical tail (combined fin and rudder) [meters**2]
            M_w -mass of wing in kg
            AR_v -aspect ratio of vertial tail
            sweep_v - sweep angle of the vertical tail [radians]
            q_c - dynamic pressure at cruise
            taper_v - taper ratio of vertical tail
            t_c_v -thickness to chord ratio of wing
            Nult - ultimate load of the aircraft [dimensionless]
            TOW - maximum takeoff weight of the aircraft [kilograms]
            S_gross_w - wing gross area [meters**2]
            t_tail - factor to determine if aircraft has a t-tail [dimensionless]
            rudder_fraction - fraction of the vertical tail that is the rudder [dimensionless]
        
        Outputs:
            output - a dictionary with outputs:
                wt_tail_vertical - weight of the vertical fin portion of the vertical tail [kilograms]
            
        Assumptions:
            uses correlations from Aircraft Design: A Conceptual Approach by Raymer
            Vertical tail weight is the weight of the vertical fin without the rudder weight.
            Rudder occupies 25% of the S_v and weighs 60% more per unit area.
   """     
    # unpack inputs
    W_0  = TOW / Units.lb # Convert kg to lbs
    S_vt  = S_v/ Units.ft**2 # Convert from meters squared to ft squared  
    q   = q_c /(Units.force_pound / Units.ft**2.)
    
    
    
    # Determine weight of the vertical portion of the tail
    if t_tail == "yes": 
        T_tail_factor = 1.# Weight of vertical portion of the T-tail is 25% more than a conventional tail
    else: 
        T_tail_factor = 0.
    
    # Calculate weight of wing for traditional aircraft vertical tail without rudder
    tail_vert_English = .073*(1+.2*T_tail_factor)*((Nult*W_0)**(.376))*(q**.122)*(S_vt**.873)*((100.*t_c_v/np.cos(sweep_v))**(-.49))*((AR_v/(np.cos(sweep_v)**2.))**.357)*(taper_v**.039)
    
    
    # packup outputs    
    
    output = Data()
    output.wt_tail_vertical = tail_vert_English * Units.lbs # Convert from lbs to kg
  
    return output
