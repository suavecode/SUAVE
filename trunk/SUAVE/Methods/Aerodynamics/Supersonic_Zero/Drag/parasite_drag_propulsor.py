# parasite_drag_propulsor.py
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


def parasite_drag_propulsor(state,settings,geometry):
#def parasite_drag_propulsor(conditions,configuration,propulsor):
    """ SUAVE.Methods.parasite_drag_propulsor(conditions,configuration,propulsor)
        computes the parasite drag associated with a propulsor 
        
        Inputs:

        Outputs:

        Assumptions:

        
    """



    # unpack inputs
    
    conditions = state.conditions
    configuration = settings
    propulsor = geometry
    
    # unpack inputs
    try:
        form_factor = configuration.propulsor_parasite_drag_form_factor
    except(AttributeError):
        form_factor = 2.3
        
    freestream = conditions.freestream
    
    Sref        = propulsor.nacelle_diameter**2 / 4 * np.pi
    Swet        = propulsor.nacelle_diameter * np.pi * propulsor.engine_length
    
    l_prop  = propulsor.engine_length
    d_prop  = propulsor.nacelle_diameter
    
    # conditions
    Mc  = freestream.mach_number
    roc = freestream.density
    muc = freestream.dynamic_viscosity
    Tc  = freestream.temperature    
    pc  = freestream.pressure

    # reynolds number
    V = Mc * compute_speed_of_sound(Tc, pc) 
    Re_prop = roc * V * l_prop/muc
    
    # skin friction coefficient
    cf_prop, k_comp, k_reyn = compressible_turbulent_flat_plate(Re_prop,Mc,Tc)
    
    # form factor for cylindrical bodies
    try: # Check if propulsor has an intake
        A_max = propulsor.nacelle_diameter
        A_exit = propulsor.A7
        A_inflow = propulsor.Ao
        d_d = 1/((propulsor.engine_length + propulsor.D) / np.sqrt(4/np.pi*(A_max - (A_exit+A_inflow)/2)))
    except:
        d_d = float(d_prop)/float(l_prop)
    D = np.array([[0.0]] * len(Mc))
    a = np.array([[0.0]] * len(Mc))
    du_max_u = np.array([[0.0]] * len(Mc))
    k_prop = np.array([[0.0]] * len(Mc))
    
    D[Mc < 0.95] = np.sqrt(1 - (1-Mc[Mc < 0.95]**2) * d_d**2)
    a[Mc < 0.95] = 2 * (1-Mc[Mc < 0.95]**2) * (d_d**2) *(np.arctanh(D[Mc < 0.95])-D[Mc < 0.95]) / (D[Mc < 0.95]**3)
    du_max_u[Mc < 0.95] = a[Mc < 0.95] / ( (2-a[Mc < 0.95]) * (1-Mc[Mc < 0.95]**2)**0.5 )
    
    D[Mc >= 0.95] = np.sqrt(1 - d_d**2)
    a[Mc >= 0.95] = 2  * (d_d**2) *(np.arctanh(D[Mc >= 0.95])-D[Mc >= 0.95]) / (D[Mc >= 0.95]**3)
    du_max_u[Mc >= 0.95] = a[Mc >= 0.95] / ( (2-a[Mc >= 0.95]) )
    
    k_prop = (1 + form_factor*du_max_u)**2
    
    # --------------------------------------------------------
    # find the final result    
    propulsor_parasite_drag = k_prop * cf_prop * Swet / Sref  
    # --------------------------------------------------------
    
    # dump data to conditions
    propulsor_result = Results(
        wetted_area               = Swet    , 
        reference_area            = Sref    , 
        parasite_drag_coefficient = propulsor_parasite_drag ,
        skin_friction_coefficient = cf_prop ,
        compressibility_factor    = k_comp  ,
        reynolds_factor           = k_reyn  , 
        form_factor               = k_prop  ,
    )
    state.conditions.aerodynamics.drag_breakdown.parasite[propulsor.tag] = propulsor_result    
    
    return propulsor_parasite_drag


# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__': 
    raise NotImplementedError
