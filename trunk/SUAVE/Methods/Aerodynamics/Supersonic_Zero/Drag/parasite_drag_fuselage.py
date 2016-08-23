# parasite_drag_fuselage.py
# 
# Created:  Aug 2014, T. Macdonald
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from compressible_turbulent_flat_plate import compressible_turbulent_flat_plate
from SUAVE.Analyses import Results

import numpy as np

# ----------------------------------------------------------------------
#   Parasite Drag Fuselage
# ----------------------------------------------------------------------

def parasite_drag_fuselage(state,settings,geometry):
    """ SUAVE.Methods.parasite_drag_fuselage(conditions,configuration,fuselage)
        computes the parasite drag associated with a fuselage 
        
        Inputs:

        Outputs:

        Assumptions:

        
    """

    # unpack inputs
    configuration = settings
    form_factor   = configuration.fuselage_parasite_drag_form_factor
    fuselage      = geometry
    
    freestream  = state.conditions.freestream
    Sref        = fuselage.areas.front_projected
    Swet        = fuselage.areas.wetted
    
    #l_fus  = fuselage.lengths.cabin
    l_fus  = fuselage.lengths.total
    d_fus  = fuselage.width
    l_nose = fuselage.lengths.nose
    l_tail = fuselage.lengths.tail
    
    # conditions
    Mc  = freestream.mach_number
    Tc  = freestream.temperature    
    re  = freestream.reynolds_number

    # reynolds number
    Re_fus = re*(l_fus + l_nose + l_tail)
    
    # skin friction coefficient
    cf_fus, k_comp, k_reyn = compressible_turbulent_flat_plate(Re_fus,Mc,Tc)
    
    # form factor for cylindrical bodies
    d_d = float(d_fus)/float(l_fus)
    D = np.array([[0.0]] * len(Mc))
    a = np.array([[0.0]] * len(Mc))
    du_max_u = np.array([[0.0]] * len(Mc))
    k_fus = np.array([[0.0]] * len(Mc))
    
    D[Mc < 0.95] = np.sqrt(1 - (1-Mc[Mc < 0.95]**2) * d_d**2)
    a[Mc < 0.95] = 2 * (1-Mc[Mc < 0.95]**2) * (d_d**2) *(np.arctanh(D[Mc < 0.95])-D[Mc < 0.95]) / (D[Mc < 0.95]**3)
    du_max_u[Mc < 0.95] = a[Mc < 0.95] / ( (2-a[Mc < 0.95]) * (1-Mc[Mc < 0.95]**2)**0.5 )
    
    D[Mc >= 0.95] = np.sqrt(1 - d_d**2)
    a[Mc >= 0.95] = 2  * (d_d**2) *(np.arctanh(D[Mc >= 0.95])-D[Mc >= 0.95]) / (D[Mc >= 0.95]**3)
    du_max_u[Mc >= 0.95] = a[Mc >= 0.95] / ( (2-a[Mc >= 0.95]) )
    
    k_fus = (1 + form_factor*du_max_u)**2

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
    try:
        state.conditions.aerodynamics.drag_breakdown.parasite[fuselage.tag] = fuselage_result
    except:
        print("Drag Polar Mode fuse parasite")
    
    return fuselage_parasite_drag