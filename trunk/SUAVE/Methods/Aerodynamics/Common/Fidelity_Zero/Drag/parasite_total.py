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
from SUAVE.Components.Energy.Networks.Lift_Cruise import Lift_Cruise
from SUAVE.Components.Energy.Networks.Battery_Propeller import Battery_Propeller
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
    geometry.networks.number_of_engines               [Unitless]
    geometry.networks.nacelle_diameter                [m]
    conditions.aerodynamics.drag_breakdown.
      parasite[wing.tag].parasite_drag_coefficient      [Unitless]
      parasite[fuselage.tag].parasite_drag_coefficient  [Unitless]
      parasite[network.tag].parasite_drag_coefficient [Unitless]


    Outputs:
    total_parasite_drag                                 [Unitless]

    Properties Used:
    N/A
    """

    # unpack
    conditions             =  state.conditions
    wings                  = geometry.wings
    fuselages              = geometry.fuselages
    networks               = geometry.networks
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

    # from networks
    for network in networks.values():
        ref_area = network.nacelle_diameter**2 / 4 * np.pi
        parasite_drag = conditions.aerodynamics.drag_breakdown.parasite[network.tag].parasite_drag_coefficient
        conditions.aerodynamics.drag_breakdown.parasite[network.tag].parasite_drag_coefficient  = parasite_drag * ref_area/vehicle_reference_area * network.number_of_engines

        # sum of parasite drag from all rotors
        if isinstance(network,Lift_Cruise) or isinstance(network,Battery_Propeller):
            if 'propellers' in network.keys():
                if np.isscalar(network.nacelle_diameter):
                    total_parasite_drag += parasite_drag * ref_area/vehicle_reference_area * network.number_of_propeller_engines
                else:
                    # sum using parasite_drag from each prop
                    total_parasite_drag += np.atleast_2d(np.sum(parasite_drag * np.atleast_2d(ref_area/vehicle_reference_area),axis=1)).T
            if 'lift_rotors' in network.keys():
                if np.isscalar(network.nacelle_diameter):
                    total_parasite_drag += parasite_drag * ref_area/vehicle_reference_area * network.number_of_lift_rotor_engines
                else:
                    # sum using parasite_drag from each lift rotor
                    total_parasite_drag += np.atleast_2d(np.sum(parasite_drag * np.atleast_2d(ref_area/vehicle_reference_area),axis=1)).T
        else:
            total_parasite_drag += parasite_drag * ref_area/vehicle_reference_area * network.number_of_engines

    # from pylons
    try:
        parasite_drag = np.atleast_2d(np.sum(conditions.aerodynamics.drag_breakdown.parasite['pylon'].parasite_drag_coefficient,axis=1)).T
    except:
        parasite_drag = 0. # not currently available for supersonics

    total_parasite_drag += parasite_drag

    # dump to condtitions
    state.conditions.aerodynamics.drag_breakdown.parasite.total = total_parasite_drag

    return total_parasite_drag
