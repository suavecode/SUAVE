# parasite_drag_wing.py
# 
# Created:  Your Name, Dec 2013
# Modified:         


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave 
from SUAVE.Methods.Aerodynamics.Supersonic_Zero.Drag import \
     parasite_drag_wing, parasite_drag_fuselage

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

def parasite_drag_aircraft(conditions,configuration,geometry):   
    """ SUAVE.Methods.parasite_drag_aircraft(aircraft,segment,Cl,cdi_inv,cdp,fd_ws)
        computes the parasite_drag_aircraft associated with an aircraft 
        
        Inputs:

        
        Outputs:

        
        Assumptions:
            based on a set of fits
        
    """

    # unpack inputs
    wings     = geometry.Wings
    fuselages = geometry.Fuselages
    vehicle_reference_area = geometry.reference_area
    drag_breakdown = conditions.aerodynamics.drag_breakdown
    
    # the drag to be returned
    total_parasite_drag = 0.0
    
    # start conditions node
    drag_breakdown.parasite = Result()
    
    # from wings
    for wing in wings.values():
        parasite_drag = parasite_drag_wing_supersonic(conditions,configuration,wing)
        conditions.aerodynamics.drag_breakdown.parasite[wing.tag].parasite_drag_coefficient = parasite_drag * wing.Areas.reference/vehicle_reference_area
        total_parasite_drag += parasite_drag * wing.Areas.reference/vehicle_reference_area
        
    # from fuselage
    for fuselage in fuselages.values():
        parasite_drag = parasite_drag_fuselage_supersonic(conditions,configuration,fuselage)
        conditions.aerodynamics.drag_breakdown.parasite[fuselage.tag].parasite_drag_coefficient = parasite_drag * fuselage.Areas.front_projected/vehicle_reference_area
        total_parasite_drag += parasite_drag * fuselage.Areas.front_projected/vehicle_reference_area
        
    # dump to condtitions
    drag_breakdown.parasite.total = total_parasite_drag
   
        
    return total_parasite_drag