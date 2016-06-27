# parasite_drag_wing.py
# 
# Created:  Aug 2014, T. Macdonald
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave 
from SUAVE.Methods.Aerodynamics.Supersonic_Zero.Drag import \
     parasite_drag_wing, parasite_drag_fuselage, parasite_drag_propulsor

from SUAVE.Analyses import Results

import numpy as np

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
    wings     = geometry.wings
    fuselages = geometry.fuselages
    propulsors = geometry.propulsors
    vehicle_reference_area = geometry.reference_area
    drag_breakdown = conditions.aerodynamics.drag_breakdown
    
    # the drag to be returned
    total_parasite_drag = 0.0
    
    # start conditions node
    drag_breakdown.parasite = Results()
    
    # from wings
    for wing in wings.values():
        parasite_drag = parasite_drag_wing(conditions,configuration,wing)
        conditions.aerodynamics.drag_breakdown.parasite[wing.tag].parasite_drag_coefficient = parasite_drag * wing.areas.reference/vehicle_reference_area
        total_parasite_drag += parasite_drag * wing.areas.reference/vehicle_reference_area
        
    # from fuselage
    for fuselage in fuselages.values():
        parasite_drag = parasite_drag_fuselage(conditions,configuration,fuselage)
        conditions.aerodynamics.drag_breakdown.parasite[fuselage.tag].parasite_drag_coefficient = parasite_drag * fuselage.areas.front_projected/vehicle_reference_area
        total_parasite_drag += parasite_drag * fuselage.areas.front_projected/vehicle_reference_area
        
    # from propulsors
    for propulsor in propulsors.values():
        parasite_drag = parasite_drag_propulsor(conditions,configuration,propulsor)
        ref_area = propulsor.nacelle_diameter**2 / 4 * np.pi
        conditions.aerodynamics.drag_breakdown.parasite[propulsor.tag].parasite_drag_coefficient = parasite_drag * ref_area/vehicle_reference_area * propulsor.number_of_engines
        total_parasite_drag += parasite_drag * ref_area/vehicle_reference_area * propulsor.number_of_engines        
        
    # dump to condtitions
    drag_breakdown.parasite.total = total_parasite_drag
   
        
    return total_parasite_drag