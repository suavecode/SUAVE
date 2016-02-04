# compressibility_drag_wing_total.py
# 
# Created:  Jan 2014, SUAVE Team
# Modified: Jan 2016, E. Botero
#        

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
# SUAVE imports
from SUAVE.Core import Results
import autograd.numpy as np 

# ----------------------------------------------------------------------
#  Computes the compressibility drag of the wings
# ----------------------------------------------------------------------

def compressibility_drag_wing_total(state,settings,geometry):
    """ SUAVE.Methods.parasite_drag_pylon(conditions,configuration,geometry):
        Simplified estimation, considering pylon drag a fraction of the nacelle drag

        Inputs:
            conditions      - data dictionary for output dump
            configuration   - not in use
            geometry        - SUave type vehicle

        Outpus:
            cd_misc  - returns the miscellaneous drag associated with the vehicle

        Assumptions:
            simplified estimation, considering pylon drag a fraction of the nacelle drag

    """

    # unpack
    conditions             = state.conditions
    wings                  = geometry.wings
    fuselages              = geometry.fuselages
    propulsors             = geometry.propulsors
    vehicle_reference_area = geometry.reference_area
    
    #compute parasite drag total
    total_compressibility_drag = 0.0
    
    # from wings
    for wing in wings.values():
        compressibility_drag = conditions.aerodynamics.drag_breakdown.compressible[wing.tag].compressibility_drag
        total_compressibility_drag += compressibility_drag 

    conditions.aerodynamics.drag_breakdown.compressible.total  = total_compressibility_drag
        
    return total_compressibility_drag
