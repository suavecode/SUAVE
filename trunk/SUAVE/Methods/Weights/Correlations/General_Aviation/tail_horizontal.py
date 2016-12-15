# tail_horizontal.py
#
# Created:  Mar 2016, M Vegh
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units
import numpy as np

# ----------------------------------------------------------------------
#   Tail Horizontal
# ----------------------------------------------------------------------

def tail_horizontal(S_h, AR_h, sweep_h, q_c, taper_h, t_c_h,Nult,TOW):       
    """ weight = SUAVE.Methods.Weights.Correlations.Tube_Wing.tail_horizontal(b_h,sweep_h,Nult,S_h,TOW,mac_w,mac_h,l_w2h,t_c_h)
        Calculate the weight of the horizontal tail in a standard configuration
        
        Inputs:
            S_h  = trapezoidal area of horizontal tail [m**2]
            Ar_h = aspect ratio of horizontal tail
            q_c  = dynamic pressure at cruise [Pa]
            
        Outputs:
            weight - weight of the horizontal tail [kilograms]
            
        Assumptions:
            calculated weight of the horizontal tail including the elevator
            Assume that the elevator is 25% of the horizontal tail
    """
    # unpack inputs
    W_0  = TOW / Units.lb # Convert kg to lbs
    S_ht  = S_h/ Units.ft**2 # Convert from meters squared to ft squared  
    q   = q_c /(Units.force_pound / Units.ft**2.)

    
    #Calculate weight of wing for traditional aircraft horizontal tail
    weight_English = .016*((Nult*W_0)**.414)*(q**.168)*(S_ht**.896)*((100.*t_c_h/np.cos(sweep_h))**(-.12))*((AR_h/(np.cos(sweep_h)**2))**.043)*(taper_h**(-.02))
    weight = weight_English * Units.lbs # Convert from lbs to kg

    return weight