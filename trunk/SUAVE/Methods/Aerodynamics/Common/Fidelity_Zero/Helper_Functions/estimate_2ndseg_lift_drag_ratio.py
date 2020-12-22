## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Helper_Functions
# estimate_2ndseg_lift_drag_ratio.py
# 
# Created:  Jun 2013, C. Ilario & T. Orra
# Modified: Oct 2015, T. Orra
#           Jan 2016, E. Botero
#           Jul 2020, E. Botero 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE Imports
from SUAVE.Components import Wings
from SUAVE.Core import Data, Units
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import compute_max_lift_coeff

# ----------------------------------------------------------------------
#  Compute 2nd segment lift to drag ratio
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Helper_Functions
def estimate_2ndseg_lift_drag_ratio(state,settings,geometry):
    """Estimates the 2nd segment climb lift to drag ratio (all engine operating)
    
    Assumptions:
    All engines operating

    Source:
    Fig. 27.34 of "Aerodynamic Design of Transport Airplane" - Obert

    Inputs:
    config.
      V2_VS_ratio              [Unitless]
      wings.
        areas.reference        [m^2]
	spans.projected        [m]
	aspect_ratio           [Unitless]
      maximum_lift_coefficient [Unitless]

    Outputs:
    lift_drag_ratio            [Unitless]

    Properties Used:
    N/A
    """
    # ==============================================
	# Unpack
    # ==============================================
    try:
        V2_VS_ratio    = geometry.V2_VS_ratio
    except:
        V2_VS_ratio    = 1.20 # typical condition

    # getting geometrical data (aspect ratio)
    n_wing = 0
    for wing in geometry.wings:
        if not isinstance(wing,Wings.Main_Wing): continue
        reference_area = wing.areas.reference
        wing_span      = wing.spans.projected
        try:
            aspect_ratio = wing.aspect_ratio
        except:
            aspect_ratio = wing_span ** 2 / reference_area
        n_wing += 1

    if n_wing > 1:
        print(' More than one Main_Wing in the config. Last one will be considered.')
    elif n_wing == 0:
        print('No Main_Wing defined! Using the 1st wing found')
        for wing in geometry.wings:
            if not isinstance(wing,Wings.Wing): continue
            reference_area = wing.areas.reference
            wing_span      = wing.spans.projected
            try:
                aspect_ratio = wing.aspect_ratio
            except:
                aspect_ratio = wing_span ** 2 / reference_area
            break


    # ==============================================
    # Determining vehicle maximum lift coefficient
    # ==============================================
    maximum_lift_coefficient, induced_drag_high_lift = compute_max_lift_coeff(state,settings,geometry)

    # Compute CL in V2
    lift_coeff = maximum_lift_coefficient / (V2_VS_ratio ** 2)

    # Estimate L/D in 2nd segment condition, ALL ENGINES OPERATIVE!
    lift_drag_ratio = -6.464 * lift_coeff + 7.264 * aspect_ratio ** 0.5

    return lift_drag_ratio