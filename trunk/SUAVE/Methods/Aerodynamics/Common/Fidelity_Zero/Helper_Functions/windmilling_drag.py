## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Helper_Functions
# windmilling_drag_coefficient.py
#
# Created:  Jul 2014, T. Orra, C. Ilario, 
# Modified: Oct 2015, T. Orra
#           Jan 2016, E. Botero  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE Imports
import SUAVE
from SUAVE.Components import Wings
from SUAVE.Core import Units, Data

# ----------------------------------------------------------------------
#  Compute drag of turbofan in windmilling condition
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Helper_Functions
def windmilling_drag(geometry,state):
    """Computes windmilling drag for turbofan engines

    Assumptions:
    None

    Source:
    http://www.dept.aoe.vt.edu/~mason/Mason_f/AskinThesis2002_13.pdf

    Inputs:
    geometry.
      max_mach_operational        [Unitless]
      reference_area              [m^2]
      wings.sref                  [m^2]
      networks.
        areas.wetted              [m^2]
        length                    [m]

    Outputs:
    windmilling_drag_coefficient  [Unitless]

    Properties Used:
    N/A
    """    
    # ==============================================
        # Unpack
    # ==============================================
    vehicle = geometry

    # Defining reference area
    if vehicle.reference_area:
        reference_area = vehicle.reference_area
    else:
        n_wing = 0
        for wing in vehicle.wings:
            if not isinstance(wing,Wings.Main_Wing): continue
            n_wing = n_wing + 1
            reference_area = wing.sref
        if n_wing > 1:
            print(' More than one Main_Wing in the vehicle. Last one will be considered.')
        elif n_wing == 0:
            print('No Main_Wing defined! Using the 1st wing found')
            for wing in vehicle.wings:
                if not isinstance(wing,Wings.Wing): continue
                reference_area = wing.sref
                break

    # getting geometric data from engine (estimating when not available)
    swet_nac = 0
    for nacelle in vehicle.nacelles: 
        swet_nac += nacelle.areas.wetted

    # Compute
    windmilling_drag_coefficient = 0.007274 * swet_nac / reference_area

    # dump data to state
    windmilling_result = Data(
        wetted_area                  = swet_nac    ,
        windmilling_drag_coefficient = windmilling_drag_coefficient ,
    )
    state.conditions.aerodynamics.drag_breakdown.windmilling_drag = windmilling_result

    return windmilling_drag_coefficient