## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Drag
# parasite_drag_total.py
#
# Created:  Jan 2014, T. Orra
# Modified: Jan 2016, E. Botero 
#           Jul 2017, M. Clarke
#           Jul 2021, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Total Parasite Drag
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Drag
def parasite_total(state,settings,geometry):
    """Sums component parasite drag

    Assumptions:
    None

    Source:
    None

    Inputs:
    geometry.reference_area                             [m^2]
    geometry.wings.areas.reference                      [m^2]
    geometry.fuselages.areas.front_projected            [m^2] 
    geometry.nacelles.diameter                          [m]
    conditions.aerodynamics.drag_breakdown.
      parasite[wing.tag].parasite_drag_coefficient      [Unitless]
      parasite[fuselage.tag].parasite_drag_coefficient  [Unitless]
      parasite[nacelle.tag].parasite_drag_coefficient   [Unitless]


    Outputs:
    total_parasite_drag                                 [Unitless]

    Properties Used:
    N/A
    """

    # unpack
    conditions             =  state.conditions
    wings                  = geometry.wings
    fuselages              = geometry.fuselages
    nacelles               = geometry.nacelles 
    vehicle_reference_area = geometry.reference_area
    
    #compute parasite drag total
    total_parasite_drag = 0.0
    
    # from wings
    for wing in wings.values():
        parasite_drag = conditions.aerodynamics.drag_breakdown.parasite[wing.tag].parasite_drag_coefficient 
        conditions.aerodynamics.drag_breakdown.parasite[wing.tag].parasite_drag_coefficient = parasite_drag * wing.areas.reference/vehicle_reference_area
        total_parasite_drag += parasite_drag * wing.areas.reference/vehicle_reference_area
 
    # from fuselage
    for fuselage in fuselages.values():
        if fuselage.tag == 'fuselage_bwb':
            continue
        parasite_drag = conditions.aerodynamics.drag_breakdown.parasite[fuselage.tag].parasite_drag_coefficient 
        conditions.aerodynamics.drag_breakdown.parasite[fuselage.tag].parasite_drag_coefficient = parasite_drag * fuselage.areas.front_projected/vehicle_reference_area
        total_parasite_drag += parasite_drag * fuselage.areas.front_projected/vehicle_reference_area
    
    # from pnacelles
    for nacelle in nacelles.values():
        ref_area = nacelle.diameter**2 / 4 * np.pi
        parasite_drag = conditions.aerodynamics.drag_breakdown.parasite[nacelle.tag].parasite_drag_coefficient 
        conditions.aerodynamics.drag_breakdown.parasite[nacelle.tag].parasite_drag_coefficient  = parasite_drag * ref_area/vehicle_reference_area
        total_parasite_drag += parasite_drag * ref_area/vehicle_reference_area 
 
    # from pylons
    try:
        parasite_drag = conditions.aerodynamics.drag_breakdown.parasite['pylon'].parasite_drag_coefficient
    except:
        parasite_drag = 0. # not currently available for supersonics

    total_parasite_drag += parasite_drag
        
    # dump to condtitions
    state.conditions.aerodynamics.drag_breakdown.parasite.total = total_parasite_drag

    return total_parasite_drag