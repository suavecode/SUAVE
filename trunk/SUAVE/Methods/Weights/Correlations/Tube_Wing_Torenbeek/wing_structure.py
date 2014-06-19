# wing_structure.py
# 
# Created:  Andrew Wendorff, May 2014      
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Attributes import Units as Units
import numpy as np
from SUAVE.Structure import (
    Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------

def wing_structure(S_gross_w,b,lambda_w,t_c_w,Nult,TakeoffWeight,eng_wing,sweep_LE,sweep_half,V_D,n_s,W_Wing,braced,S_f,b_f,V_stall,d_f,sweep_f,t_c_f,flap_type,variable_flap,undercarriage):
    """ weight = SUAVE.Methods.Weights.Correlations.Tube_Wing.wing_structure\
        (S_gross_w,b,lambda_w,t_c_w,Nult,TOW,eng_wing,sweep_LE,sweep_half,V_D,\
        n_s,W_Wing,braced,S_f,b_f,V_stall,d_f,sweep_f,t_c_f,flap_type,variable_flap,undercarriage)
        Calculate the wing weight of the aircraft based on the fully-stressed 
        bending weight of the wing box        
        
        Inputs:
            S_gross_w - area of the wing [meters**2]
            b - span of the wing [meters**2]
            lambda_w - taper ratio of the wing [dimensionless]
            t_c_w - thickness-to-chord ratio of the wing [dimensionless]
            Nult - ultimate load factor of the aircraft [dimensionless]
            TakeoffWeight - maximum takeoff weight of the aircraft [kilograms]
            eng_wing - number of wing-attached engines [dimensionless]
            sweep_LE - leading edge sweep of the wing [radians]
            sweep_half - half-chord sweep of the wing [radians]
            V_D - Maximum Design Diving Speed [m/s]
            n_s - Distance of the strut mounting on a braced wing from a wing root divided by semi-chord span [dimensionless]
            W_Wing - Wing group weight [kilograms] 
            braced - Specifies in the wing is "braced" or "not braced"
            S_f - Area of the flaps [meters ** 2]
            b_f - Structural flap span of the wing [meters]
            V_stall - Stall speed of the aircraft in the landing configuration [meters/sec]
            d_f - Maximum angular deflection of the flaps  [radians]
            sweep_f - Average sweep angle of the flap structure
            t_c_f - Flap thickness-to-chord ratio in the streamwise direction
            flap_type - Denoting the flap type of the aircraft: "single-slotted, "double slotted, fixed hinge", "double slotted, 4-bar movement"\
                "single slotted Fowler", "double slotted Fowler", "triple slotted Fowler"
            variable_flap - Denoting whether there is variable geometry "yes" or no
            undercarriage - Denoting whether the undercarriage is "wing-mounted" or not mounted to the wing
            
        Outputs:
            output -
                wt_wing_structure - Weight of the main structure [kilograms]
                wt_wing_high_lift - Weight of the high lift structure [kilograms]
                wt_wing - weight of the wing including main structure, spoilers, and high lift structure[kilograms]
                
        Assumptions [Notes from Torenbeek]:
        This structural weight is derived from the requirement that the critical flight condition to be resisted is the bending moment due to wing lift.
        The weight of the high-lift devices is based on a critical loading condition at the flap design speed.
        Looking at subsonic short-haul aircraft, the original method underestimates wing weight since extra weight is required for adequate stiffness for wing \
            flutter and a weight penalty is necessary due to long service life requirements. A modification to incorporate high lift devices and spoilers was then added.
        
        
        Taken from Torenbeek, "Syntheseis of Subsonic Airplace Design" Delft University Press, 1977, p 451-455.
            
    """
    
    # Unpack inputs
    V_lf = 1.8 * V_stall # Design speed for flaps in the landing configuration
    
    # Process
    k_no = 1. + (1.905/(b*(np.cos(sweep_half))**-1))*0.5 # Weight penality due to skin joints, non-tapered skin, minimum gauge, etc.
    k_lambda = (1.+lambda_w)**0.4
    if eng_wing == 2.:
        k_e = 0.95 # bending moment relief due to engine and nacelle installation
        k_st = 1. + 9.06*10.**-4. * (b * np.cos(sweep_LE))**3 *(V_D/100/t_c_w)**2 * np.cos(sweep_half) / TOW 
    elif eng_wing == 4:
        k_e = 0.9
    else:
        k_e = 1.
        k_st = 1. + 9.06*10.**-4. * (b * np.cos(sweep_LE))**3 *(V_D/100/t_c_w)**2 * np.cos(sweep_half) / TOW
        
    if undercarriage == "wing-mounted":
        k_uc = 1.    
    else: 
        k_uc = 0.95
    
    if braced == "braced":    
        k_b = 1. - n_s ** 2.
    else:
        k_b = 1.
    
    if variable_flap == "yes":
        kf_2 = 1.25
    else:
        kf_2 = 1
    
    if flap_type == "triple slotted Fowler":
        kf_1 = 1.45
    elif flap_type == "double slotted Fowler":
        kf_1 = 1.3
    elif flap_type == "single slotted Fowler" or flap_type == "double slotted,4-bar movement":
        kf_1 = 1.15
    elif flap_type == "double slotted, fixed hinge":
        kf_1 = 1.
    else:
        kf_1 = 1. 
    
    k_f = kf_1 * kf_2
    
    structure = 4.58*10**-3 * k_no * k_lambda * k_e * k_uc * k_st * (k_b * Nult * (TakeoffWeight - 0.8 * W_Wing)) ** 0.55 * b ** 1.675 * t_c_w ** -0.45 * np.cos(sweep_half) ** -1.325  
    trailing_edge = (2.706 * k_f * (S_f * b_f) ** (3./16.) * ( (V_lf/100)**2. * np.sin(d_f)* np.cos(sweep_f)/t_c_f) ** (3./4.)) * S_f
    leading_edge = 0 # Needs to be regressed from figure on p 454
    high_lift = trailing_edge + leading_edge
    spoiler = 0.015 * W_Wing
    total = structure + 1.2 * (high_lift + spoiler)
    # packup outputs
    output = Data()
    output.wt_wing_structure = structure
    output.wt_wing_high_lift = high_lift
    output.wt_wing = total # Total wing weight of the aircraft
    
    return output