# trapezoid_ac_x.py

# Created: Tim Momose, March 2014

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.trapezoid_mac import trapezoid_mac
from SUAVE.Attributes import Units as Units
from SUAVE.Structure import (
    Data, Container, Data_Exception, Data_Warning,
)


def trapezoid_ac_x(wing):
    """ dx_ac = SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.trapezoid_ac_x(wing)
        This method computes the aerodynamic center x-position of a linearly
        tapered trapezoidal aerodynamic surface

        Inputs:
            wing - a data dictionary with the fields:
                areas.reference - planform area of the trapezoidal wing [meters**2]
                spans.projected - wing span [meters]
                chords.root - wing root chord [meters]
                taper - wing taper ratio [dimensionless]
                sweep - wing leading edge sweep [radians]
                symmetric - wing symmetry [Boolean]

        Outputs:
            dx_ac - the x-direction distance of the aerodynamic center of the wing
            (or equivalent trapezoid) measured from the leading edge of the wing
            root [meters]

        Assumptions:
            Assumes a simple trapezoidal wing shape.
            Does not account for twist or LE/TE extensions
            Assumes the aerodynamic center of the wing is located at the quarter-
            chord of the MAC.
    """

    #Unpack inputs
    S     = wing.areas.reference
    b     = wing.spans.projected
    l     = wing.taper
    sweep = wing.sweep
    symm  = wing.symmetric
    c_r   = wing.chords.root

    #Get MAC
    mac = trapezoid_mac(wing)

    #Find spanwise location of MAC
    if l != 1.0:
        if not c_r:
            mgc = S/b               #mean geometric chord
            c_r = mgc/(1-0.5*(1-l)) #root chord
        mac_semispan = (b*(1-0.5*symm))*(c_r-mac)/(c_r*(1-l))
    else:
        mac_semispan = 0.5 * b

    dx_ac = mac_semispan*np.tan(sweep) + mac/4.0

    return dx_ac
