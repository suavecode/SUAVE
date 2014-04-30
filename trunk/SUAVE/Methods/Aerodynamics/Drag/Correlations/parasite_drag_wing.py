# parasite_drag_wing.py
# 
# Created:  Your Name, Dec 2013
# Modified:         

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# local imports
from compressible_turbulent_flat_plate import compressible_turbulent_flat_plate

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
            
        Outputs:
        
        Assumptions:
        
    """
    
    # unpack inputs
    C = configuration.wing_parasite_drag_form_factor
    Sref = wing.sref
    
    # wing
    mac_w        = wing.chord_mac
    t_c_w        = wing.t_c
    sweep_w      = wing.sweep
    arw_w        = wing.ar
    span_w       = wing.span    
    S_exposed_w  = wing.S_exposed # TODO: calculate by fuselage diameter (in Fidelity_Zero.initialize())
    S_affected_w = wing.S_affected    
    
    # compute wetted area # TODO: calcualte as preprocessing
    Swet = 2. * (1.0+ 0.2*t_c_w) * S_exposed_w    
    
    # conditions
    Mc  = conditions.mach_number
    roc = conditions.density
    muc = conditions.viscosity
    Tc  = conditions.temperature    
    pc  = conditions.pressure
    
    # reynolds number
    V    = Mc * compute_speed_of_sound( Tc, pc ) #input gamma and R
    Re_w = roc * V * mac_w/muc    
    
    # skin friction  coefficient
    cf_w, k_comp, k_reyn = compressible_turbulent_flat_plate(Re_w,Mc,Tc)

    # correction for airfoils
    k_w = 1. + ( 2.* C * (t_c_w * (np.cos(sweep_w))**2.) ) / ( np.sqrt(1.- Mc**2. * ( np.cos(sweep_w))**2.) )  \
             + ( C**2. * (np.cos(sweep_w))**2. * t_c_w**2. * (1. + 5.*(np.cos(sweep_w)**2.)) ) \
                / (2.*(1.-(Mc*np.cos(sweep_w))**2.))       
    
    # --------------------------------------------------------
    # find the final result
    wing_parasite_drag = k_w * cf_w * Swet / Sref 
    # --------------------------------------------------------
    
    # dump data to conditions
    wing_result = Result(
        wetted_area               = Swet   , 
        reference_area            = Sref   , 
        parasite_drag_coefficient = wing_parasite_drag ,
        skin_friction_coefficient = cf_w   ,
        compressibility_factor    = k_comp ,
        reynolds_factor           = k_reyn , 
        form_factor               = k_w    ,
    )
    #conditions.drag_breakdown.parasite[wing.tag] = wing_result
    
    # done!
    return wing_parasite_drag


# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__': 
    raise NotImplementedError
