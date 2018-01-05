# tube.py
#
# Created:  Mar 2016, M. Vegh
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units
import numpy as np

# ----------------------------------------------------------------------
#   Fuselage
# ----------------------------------------------------------------------

def fuselage(S_fus, Nult, TOW, w_fus, h_fus, l_fus, l_ht, q_c, V_fuse, diff_p_fus):
    """ weight = SUAVE.Methods.Weights.Correlations.Tube_Wing.tube(S_fus, diff_p_fus, w_fus, h_fus, l_fus, Nlim, wt_zf, wt_wing, wt_propulsion, wing_c_r)
        Calculate the weight of a fuselage in the state tube and wing configuration
        
        Inputs:
            S_f - fuselage wetted area [meters**2]
            Nult - ultimate load of the aircraft [dimensionless]]
            TOW - maximum takeoff weight of the aircraft [kilograms]
            w_fus - width of the fuselage [meters]
            h_fus - height of the fuselage [meters]
            l_fus - length of the fuselage [meters]
            V_fuse - volume of pressurized cabin [meters**3]
            diff_p_fus - Maximum fuselage pressure differential [Pascal]
           

            
        Outputs:
            weight - weight of the fuselage [kilograms]
            
        Assumptions:
            fuselage in a standard wing and tube configuration 
    """
    # unpack inputs
    
    d_fus    = (h_fus+w_fus)/2. #take average as diameter
    d_str    = .025*d_fus+1.*Units.inches   #obtained from http://homepage.ntlworld.com/marc.barbour/structures.html
    diff_p   = diff_p_fus / (Units.force_pound / Units.ft**2.) # Convert Pascals to lbs/ square ft
    width    = w_fus / Units.ft # Convert meters to ft
    height   = h_fus / Units.ft # Convert meters to ft
    tail_arm = np.abs(l_ht)/Units.ft
    V_p      = V_fuse/(Units.ft**3)
    length   = l_fus / Units.ft  # Convert meters to ft
    weight   = TOW / Units.lb    # Convert kg to lbs
    area     = S_fus / (Units.ft**2.) # Convert square meters to square ft 
    q        = q_c /(Units.force_pound / Units.ft**2.)
  

  
    #Calculate weight of wing for traditional aircraft vertical tail without rudder
    fuselage_weight = .052*(area**1.086)*((Nult*weight)**.177)*(tail_arm**(-.051))*((l_fus/d_str)**(-.072))*(q**.241)+11.9*((V_fuse* diff_p)**.271)
    
    weight = fuselage_weight*Units.lbs #convert to kg
    return weight