# parasite_drag_propulsor.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Jan 2016, E. Botero          

#Sources: Stanford AA241 Course Notes
#         Raymer: Aircraft Design: A Conceptual Approach

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Core import Data
from SUAVE.Analyses import Results
from compressible_turbulent_flat_plate import compressible_turbulent_flat_plate

# package imports
import numpy as np

# ----------------------------------------------------------------------
#   Parasite Drag Propulsor
# ----------------------------------------------------------------------

def parasite_drag_propulsor(state,settings,geometry):
    """ SUAVE.Methods.parasite_drag_propulsor(conditions,configuration,propulsor)
        computes the parasite drag associated with a propulsor 
        
        Inputs:

        Outputs:

        Assumptions:

        
    """

    # unpack inputs
    conditions    = state.conditions
    configuration = settings
    
    propulsor = geometry
    Sref      = propulsor.nacelle_diameter**2. / 4. * np.pi
    Swet      = propulsor.areas.wetted
    
    l_prop = propulsor.engine_length
    d_prop = propulsor.nacelle_diameter
    
    # conditions
    freestream = conditions.freestream
    Mc  = freestream.mach_number
    Tc  = freestream.temperature    
    re  = freestream.reynolds_number

    # reynolds number
    Re_prop = re*l_prop
    
    # skin friction coefficient
    cf_prop, k_comp, k_reyn = compressible_turbulent_flat_plate(Re_prop,Mc,Tc)
    
    ## form factor according to Raymer equation (pg 283 of Aircraft Design: A Conceptual Approach)
    k_prop = 1 + 0.35 / (float(l_prop)/float(d_prop))  
    
   
    # find the final result    
    propulsor_parasite_drag = k_prop * cf_prop * Swet / Sref
    
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
    conditions.aerodynamics.drag_breakdown.parasite[propulsor.tag] = propulsor_result    
    
    return propulsor_parasite_drag