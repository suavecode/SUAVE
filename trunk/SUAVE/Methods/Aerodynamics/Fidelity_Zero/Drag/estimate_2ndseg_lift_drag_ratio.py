# estimate_2ndseg_lift_drag_ratio.py
# 
# Created:  Jun 2013, C. Ilario & T. Orra
# Modified: Oct 2015, T. Orra
#           Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE Imports
from SUAVE.Components import Wings
from SUAVE.Core import Data, Units

# ----------------------------------------------------------------------
#  Compute 2nd segment lift to drag ratio
# ----------------------------------------------------------------------

def estimate_2ndseg_lift_drag_ratio(config):
    """ SUAVE.Methods.Aerodynamics.Drag.Correlations.estimate_2ndseg_lift_drag_ratio(config):
        Estimates the 2nd segment Lift to drag ration (all engine operating)
        Inputs:
            config   - data dictionary with fields:
                reference_area             - Airplane reference area
                V2_VS_ratio                - Ratio between V2 and Stall speed
                                             [optional. Default value = 1.20]
                Main_Wing.aspect_ratio     - Main_Wing aspect ratio
                maximum_lift_coefficient   - Maximum lift coefficient for the config
                                             [Calculated if not informed]

        Outputs:
            takeoff_field_length            - Takeoff field length

        Assumptions:
            All engines Operating. Must be corrected for second segment climb configuration
            reference: Fig. 27.34 of "Aerodynamic Design of Transport Airplane" - Obert

"""
    # ==============================================
	# Unpack
    # ==============================================
    try:
        V2_VS_ratio    = config.V2_VS_ratio
    except:
        V2_VS_ratio    = 1.20 # typical condition

    # getting geometrical data (aspect ratio)
    n_wing = 0
    for wing in config.wings:
        if not isinstance(wing,Wings.Main_Wing): continue
        reference_area = wing.areas.reference
        wing_span      = wing.spans.projected
        try:
            aspect_ratio = wing.aspect_ratio
        except:
            aspect_ratio = wing_span ** 2 / reference_area
        n_wing += 1

    if n_wing > 1:
        print ' More than one Main_Wing in the config. Last one will be considered.'
    elif n_wing == 0:
        print  'No Main_Wing defined! Using the 1st wing found'
        for wing in config.wings:
            if not isinstance(wing,Wings.Wing): continue
            reference_area = wing.areas.reference
            wing_span      = wing.spans.projected
            try:
                aspect_ratio = wing.aspect_ratio
            except:
                aspect_ratio = wing_span ** 2 / reference_area
            break


    # Determining vehicle maximum lift coefficient
    try:   # aircraft maximum lift informed by user
        maximum_lift_coefficient = config.maximum_lift_coefficient
    except:
        # Using semi-empirical method for maximum lift coefficient calculation
        from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import compute_max_lift_coeff

        # Condition to CLmax calculation: 90KTAS @ 10000ft, ISA
        conditions = Data()
        conditions.freestream = Data()
        conditions.freestream.density           = 0.90477283
        conditions.freestream.dynamic_viscosity = 1.69220918e-05
        conditions.freestream.velocity          = 90. * Units.knots
        try:
            maximum_lift_coefficient, induced_drag_high_lift = compute_max_lift_coeff(config,conditions)
            config.maximum_lift_coefficient = maximum_lift_coefficient
        except:
            raise ValueError, "Maximum lift coefficient calculation error. Please, check inputs"

    # Compute CL in V2
    lift_coeff = maximum_lift_coefficient / (V2_VS_ratio ** 2)

    # Estimate L/D in 2nd segment condition, ALL ENGINES OPERATIVE!
    lift_drag_ratio = -6.464 * lift_coeff + 7.264 * aspect_ratio ** 0.5

    return lift_drag_ratio