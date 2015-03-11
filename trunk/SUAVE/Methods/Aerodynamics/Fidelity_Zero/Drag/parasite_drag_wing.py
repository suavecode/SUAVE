# parasite_drag_wing.py
# 
# Created:  Your Name, Dec 2013
# Modified:         

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# local imports
from compressible_turbulent_flat_plate import compressible_turbulent_flat_plate
from compressible_mixed_flat_plate import compressible_mixed_flat_plate

# suave imports
from SUAVE.Attributes.Gases import Air # you should let the user pass this as input
air = Air()
compute_speed_of_sound = air.compute_speed_of_sound

from SUAVE.Attributes.Results import Result

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp


# ----------------------------------------------------------------------
#   The Function
# ----------------------------------------------------------------------

def parasite_drag_wing(conditions,configuration,wing):
    """ SUAVE.Methods.parasite_drag_wing(conditions,configuration,wing)
        computes the parastite drag associated with a wing 
        
        Inputs:
            conditions
            -freestream mach number
            -freestream density
            -freestream viscosity
            -freestream temperature
            -freestream pressuve
            
            configuration
            -wing parasite drag form factor
            
            wing
            -S reference
            -mean aerodynamic chord
            -thickness to chord ratio
            -sweep
            -aspect ratio
            -span
            -S exposed
            -S affected
            -transition x
            
        Outputs:
            wing parasite drag coefficient with refernce area as the
            reference area of the input wing
        
        Assumptions:
        
    """
    
    # unpack inputs
    C = configuration.wing_parasite_drag_form_factor
    freestream = conditions.freestream
    Sref = wing.areas.reference
    
    # wing
    mac_w        = wing.chords.mean_aerodynamic
    t_c_w        = wing.thickness_to_chord
    sweep_w      = wing.sweep
    arw_w        = wing.aspect_ratio
    span_w       = wing.spans.projected
    S_exposed_w  = wing.areas.exposed # TODO: calculate by fuselage diameter (in Fidelity_Zero.initialize())
    S_affected_w = wing.areas.affected  
    xtu          = wing.transition_x_upper
    xtl          = wing.transition_x_lower
    
    # compute wetted area 
    try:
        Swet = wing.areas.wetted
    except:
        Swet = 1. * (1.0+ 0.2*t_c_w) * S_exposed_w
        wing.areas.wetted = Swet
    
    # conditions
    Mc  = freestream.mach_number
    roc = freestream.density
    muc = freestream.viscosity
    Tc  = freestream.temperature    
    pc  = freestream.pressure
    
    # reynolds number
    V    = Mc * compute_speed_of_sound( Tc, pc ) #input gamma and R
    Re_w = roc * V * mac_w/muc    
    
    # skin friction  coefficient, upper
    cf_w_u, k_comp_u, k_reyn_u = compressible_mixed_flat_plate(Re_w,Mc,Tc,xtu)
    
    # skin friction  coefficient, lower
    cf_w_l, k_comp_l, k_reyn_l = compressible_mixed_flat_plate(Re_w,Mc,Tc,xtl)    

    # correction for airfoils
    k_w = 1. + ( 2.* C * (t_c_w * (np.cos(sweep_w))**2.) ) / ( np.sqrt(1.- Mc**2. * ( np.cos(sweep_w))**2.) )  \
             + ( C**2. * (np.cos(sweep_w))**2. * t_c_w**2. * (1. + 5.*(np.cos(sweep_w)**2.)) ) \
                / (2.*(1.-(Mc*np.cos(sweep_w))**2.))       
    
    # --------------------------------------------------------
    # find the final result
    wing_parasite_drag = k_w * cf_w_u * Swet / Sref /2. + k_w * cf_w_l * Swet / Sref /2.
    # --------------------------------------------------------
    
    # dump data to conditions
    wing_result = Result(
        wetted_area               = Swet   , 
        reference_area            = Sref   , 
        parasite_drag_coefficient = wing_parasite_drag ,
        skin_friction_coefficient = (cf_w_u+cf_w_l)/2.   ,
        compressibility_factor    = k_comp_u ,
        reynolds_factor           = k_reyn_l , 
        form_factor               = k_w    ,
    )
    conditions.aerodynamics.drag_breakdown.parasite[wing.tag] = wing_result
    
    # done!
    return wing_parasite_drag


# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__': 
    raise NotImplementedError
