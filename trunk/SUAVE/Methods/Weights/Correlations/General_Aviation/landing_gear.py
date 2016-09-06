# landing_gear.py
# 
# Created:  Mar 2016, M. Vegh
# Modified: 


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data,Units
import numpy as np





# ----------------------------------------------------------------------
#   Landing Gear
# ----------------------------------------------------------------------

def landing_gear(landing_weight, Nult, strut_length_main, strut_length_nose):
    """ weight = SUAVE.Methods.Weights.Correlations.Tube_Wing.landing_gear(TOW)
        Calculate the weight of the landing gear assuming that the gear 
        weight is 4 percent of the takeoff weight        
        
        Inputs:
            Nult - ultimate landing load factor
            landing_weight- landing weight of the aircraft [kilograms]
           
        Outputs:
            weight - weight of the landing gear [kilograms]
            
        Assumptions:
            calculating the landing gear weight based on the landing weight, load factor, and strut length 
    """ 
    
    #unpack
    
    #process
    W_l = landing_weight/Units.lbs
    l_n = strut_length_nose/Units.inches
    l_m = strut_length_main/Units.inches
    main_weight = .095*((Nult*W_l)**.768)*(l_m/12.)**.409
    nose_weight = .125*((Nult*W_l)**.566)*(l_n/12.)**.845
    
    #pack outputs
    output = Data
    output.main = main_weight*Units.lbs
    output.nose = nose_weight*Units.lbs
    
    return output