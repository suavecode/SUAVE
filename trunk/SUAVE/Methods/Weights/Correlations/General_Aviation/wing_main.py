# tail_vertical.py
#
# Created:  Mar 2016, M. Vegh
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units
import numpy as np

# ----------------------------------------------------------------------
#   Tail Vertical
# ----------------------------------------------------------------------

def wing_main(S_wing, m_fuel, AR_w, sweep_w, q_c, taper_w, t_c_w,Nult,TOW):      
    """ output = SUAVE.Methods.Weights.Correlations.GA.Raymer.wing_main(S_w, M_w, AR_w, sweep_w, q_c, taper_w, t_c_w,Nult,TOW)
        Calculate the weight of the main wing of an aircraft     
        
        Inputs:
            S_wing- area of the main wing[meters**2]
            m_fuel - predicted weight of fuel in the wing [kilograms]
            AR_w -aspect ratio of main wing
            sweep_w - quarter chord sweep of the main wing
            q_c - dynamic pressure at cruise [N/m**2]
            taper_w - taper ratio of wing
            t_c_w -thickness to chord ratio of wing
            Nult - ultimate load of the aircraft [dimensionless]
            TOW - maximum takeoff weight of the aircraft [kilograms]
   
        Outputs:
            output - a dictionary with outputs:
                wt_main_wing - weight of the vertical fin portion of the vertical tail [kilograms]
            
        Assumptions:
            uses correlations from Aircraft Design: A Conceptual Approach by Raymer
    """     
    # unpack inputs


    W_0  = TOW / Units.lb # Convert kg to lbs
    S_w  = S_wing/ (Units.ft**2) # Convert from meters squared to ft squared  
    W_fw = m_fuel/Units.lbs #convert from kg to lbs
    q   = q_c /(Units.lbs/(Units.ft**2.))

   
    # Calculate weight of wing for traditional aircraft vertical tail without rudder
    weight_English = .036 * (S_w**.758)*(W_fw**.0035)*((AR_w/(np.cos(sweep_w)**2))**.6)*(q**.006)*(taper_w**.04)*((100.*t_c_w/np.cos(sweep_w))**(-.3))*((Nult*W_0)**.49)
    # packup outputs    
    
  
    weight =  weight_English * Units.lbs # Convert from lbs to kg

    return weight
