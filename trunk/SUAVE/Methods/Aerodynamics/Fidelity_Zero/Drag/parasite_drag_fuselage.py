# parasite_drag_fuselage.py
# 
# Created:  Your Name, Dec 2013
# Modified:         

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# local imports
from compressible_turbulent_flat_plate import compressible_turbulent_flat_plate

# suave imports

from compressible_turbulent_flat_plate import compressible_turbulent_flat_plate

from SUAVE.Attributes.Gases import Air # you should let the user pass this as input
from SUAVE.Core import Results
air = Air()
compute_speed_of_sound = air.compute_speed_of_sound

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


def parasite_drag_fuselage(state,settings,geometry):
#def parasite_drag_fuselage(conditions,configuration,fuselage):
    """ SUAVE.Methods.parasite_drag_fuselage(conditions,configuration,fuselage)
        computes the parasite drag associated with a fuselage 
        
        Inputs:

        Outputs:

        Assumptions:

        
    """

    # unpack inputs
    
    conditions = state.conditions   
    configuration =settings 
    #fuselages = geometry.fuselages
    fuselage = geometry
    
    
    form_factor = configuration.fuselage_parasite_drag_form_factor
    freestream = conditions.freestream
    
    fuselage_parasite_drag_total = 0.0
    #for fuselage in fuselages.values():
    
    
    Sref        = fuselage.areas.front_projected
    Swet        = fuselage.areas.wetted
    
    l_fus  = fuselage.lengths.cabin
    d_fus  = fuselage.effective_diameter
    l_nose = fuselage.lengths.nose
    l_tail = fuselage.lengths.tail
    
    # conditions
    Mc  = freestream.mach_number
    roc = freestream.density
    muc = freestream.dynamic_viscosity
    Tc  = freestream.temperature    
    pc  = freestream.pressure

    # reynolds number
    V = Mc * compute_speed_of_sound(Tc, pc) 
    Re_fus = roc * V * (l_fus + l_nose + l_tail)/muc
    
    # skin friction coefficient
    cf_fus, k_comp, k_reyn = compressible_turbulent_flat_plate(Re_fus,Mc,Tc)
    
    # form factor for cylindrical bodies
    d_d = float(d_fus)/float(l_fus)
    D = np.sqrt(1 - (1-Mc**2) * d_d**2)
    a        = 2 * (1-Mc**2) * (d_d**2) *(np.arctanh(D)-D) / (D**3)
    du_max_u = a / ( (2-a) * (1-Mc**2)**0.5 )
    k_fus    = (1 + form_factor*du_max_u)**2
    
    # --------------------------------------------------------
    # find the final result    
    fuselage_parasite_drag = k_fus * cf_fus * Swet / Sref 
    
          
            
    # dump data to conditions
    fuselage_result = Results(
        wetted_area               = Swet   , 
        reference_area            = Sref   , 
        parasite_drag_coefficient = fuselage_parasite_drag ,
        skin_friction_coefficient = cf_fus ,
        compressibility_factor    = k_comp ,
        reynolds_factor           = k_reyn , 
        form_factor               = k_fus  ,
    )
    state.conditions.aerodynamics.drag_breakdown.parasite[fuselage.tag] = fuselage_result    
    
    return fuselage_parasite_drag


# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__': 
    raise NotImplementedError
