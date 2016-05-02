# compute_flap_lift.py
#
# Created:  Dec 2013, A. Varyar
# Modified: Feb 2014, T. Orra
#           Jan 2016, E. Botero  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units
import numpy as np

# ----------------------------------------------------------------------
#  compute_flap_lift
# ----------------------------------------------------------------------
def compute_flap_lift(t_c,flap_type,flap_chord,flap_angle,sweep,wing_Sref,wing_affected_area):
    """ SUAVE.Methods.Aerodynamics.compute_flap_lift(vehicle):
        Computes the increase of lift due to trailing edge flap deployment

        Inputs:
            t_c                 - wing thickness ratio
            flap_c_chord        - flap chord as fraction of wing chord
            flap_angle          - flap deflection               - [rad]
            sweep               - Wing sweep angle              - [rad]
            wing_Sref           - Wing reference area           - [m?]
            wing_affected_area  - Wing area affected by flaps   - [m?]
                                  NOTE.: do not confuse with flap area

        Outputs:
            dcl_flap    - Lift coefficient increase due to trailing edge flap

        Assumptions:
            if needed

    """

    #unpack
    tc_r  = t_c
    fc    = flap_chord * 100.
    fa    = flap_angle / Units.deg
    Swf   = wing_affected_area
    sweep = sweep

    # Basic increase in CL due to flap
    dmax_ref= -4E-05*tc_r**4 + 0.0014*tc_r**3 - 0.0093*tc_r**2 + 0.0436*tc_r + 0.9734

    # Corrections for flap type
    if flap_type == 'none':
        dmax_ref = 0.
    if flap_type == 'single_slotted':
        dmax_ref = dmax_ref * 0.93
    if flap_type == 'triple_slotted':
        dmax_ref = dmax_ref * 1.08

    # Chord correction
    Kc =  0.0395*fc    + 0.0057

    # Deflection correction
    Kd = -1.7857E-04*fa**2 + 2.9214E-02*fa - 1.4000E-02

    # Sweep correction
    Ksw = (1 - 0.08 * (np.cos(sweep))**2) * (np.cos(sweep)) ** 0.75

    # Applying corrections
    dmax_flaps = Kc * Kd * Ksw * dmax_ref

    # Final CL increment due to flap
    dcl_max_flaps = dmax_flaps  *  Swf / wing_Sref

    return dcl_max_flaps


# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':

    #imports
    import SUAVE
    from SUAVE.Core import Units

    # Test case
    t_c             = 0.11
    flap_type       = 'single_slotted'
    flap_chord      = 0.28
    flap_angle      = 30. * Units.deg
    sweep           = 30. * Units.deg
    wing_Sref       = 120.
    wing_flap_area  = 120. * .6

    dcl_flap = compute_flap_lift(t_c,flap_type,flap_chord,flap_angle,sweep,wing_Sref,wing_flap_area)
    print 'Delta CL due to Flaps: ', dcl_flap

    # Test case
    t_c             = 0.11
    flap_type       = 'none'
    flap_chord      = 0.
    flap_angle      = 0. * Units.deg
    sweep           = 25. * Units.deg
    wing_Sref       = 120.
    wing_flap_area  = 0.

    dcl_flap = compute_flap_lift(t_c,flap_type,flap_chord,flap_angle,sweep,wing_Sref,wing_flap_area)
    print 'Delta CL due to Flaps: ', dcl_flap