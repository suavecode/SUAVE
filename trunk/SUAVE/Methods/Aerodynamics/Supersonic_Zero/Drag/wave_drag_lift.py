## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
# wave_drag_lift.py
# 
# Created:  Feb 2019, T. MacDonald
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Data
from SUAVE.Components.Wings import Main_Wing

# ----------------------------------------------------------------------
#   Wave Drag Lift
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
def wave_drag_lift(conditions,configuration,wing):
    """Computes wave drag due to lift

    Assumptions:
    Simplified equations

    Source:
    http://aerodesign.stanford.edu/aircraftdesign/drag/ssdragcalc.html

    Inputs:
    conditions.freestream.mach_number        [Unitless]
    conditions.aerodynamics.lift_coefficient [Unitless]
    wing.total_length                        [m]
    wing.areas.reference                     [m^2]

    Outputs:
    wave_drag_lift                           [Unitless]

    Properties Used:
    N/A
    """  

    # Unpack
    freestream   = conditions.freestream
    total_length = wing.total_length
    Sref         = wing.areas.reference
    
    # Conditions
    Mc  = freestream.mach_number * 1.0

    # Length-wise aspect ratio
    ARL = total_length**2/Sref
    
    # Lift coefficient
    if isinstance(wing,Main_Wing):
        CL = conditions.aerodynamics.lift_coefficient
    else:
        CL = np.zeros_like(conditions.aerodynamics.lift_coefficient)
    
    mach_ind = Mc >= 1.01
    
    # Computations
    x = np.pi*ARL/4
    beta = np.array([[0.0]] * len(Mc))
    beta[Mc >= 1.01] = np.sqrt(Mc[Mc >= 1.01]**2-1)
    wave_drag_lift = np.array([[0.0]] * len(Mc))
    wave_drag_lift[Mc >= 1.01] = CL[Mc >= 1.01]**2*x/4*(np.sqrt(1+(beta[Mc >= 1.01]/x)**2)-1)
    wave_drag_lift[0:len(Mc[Mc >= 1.01]),0] = wave_drag_lift[Mc >= 1.01]
    
    # Dump data to conditions
    wave_lift_result = Data(
        reference_area             = Sref   , 
        wave_drag_lift_coefficient = wave_drag_lift ,
        length_AR                  = ARL,
    )

    return wave_drag_lift
