# wave_drag_lift.py
# 
# Created:  Jun 2014, T. Macdonald
# Modified: May 2017, T. Macdonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Analyses import Results

# ----------------------------------------------------------------------
#   Wave Drag Lift
# ----------------------------------------------------------------------

def wave_drag_lift(conditions,configuration,wing):

    # Unpack
    freestream = conditions.freestream
    total_length = wing.total_length
    Sref = wing.areas.reference
    
    # Conditions
    Mc  = freestream.mach_number * 1.0

    # Length-wise aspect ratio
    ARL = total_length**2/Sref
    
    if wing.vertical:
        CL = np.zeros_like(conditions.aerodynamics.lift_coefficient)
    else:
        CL = conditions.aerodynamics.lift_breakdown.inviscid_wings_lift[wing.tag]
    
    # Computations
    x = np.pi*ARL/4
    beta = np.array([[0.0]] * len(Mc))
    beta[Mc >= 1.05] = np.sqrt(Mc[Mc >= 1.05]**2-1)
    wave_drag_lift = np.array([[0.0]] * len(Mc))
    wave_drag_lift[Mc >= 1.05] = CL[Mc >= 1.05]**2*x/4*(np.sqrt(1+(beta[Mc >= 1.05]/x)**2)-1)
    wave_drag_lift[0:len(Mc[Mc >= 1.05]),0] = wave_drag_lift[Mc >= 1.05]
    
    # Dump data to conditions
    wave_lift_result = Results(
        reference_area             = Sref   , 
        wave_drag_lift_coefficient = wave_drag_lift ,
        length_AR                  = ARL,
    )

    return wave_drag_lift