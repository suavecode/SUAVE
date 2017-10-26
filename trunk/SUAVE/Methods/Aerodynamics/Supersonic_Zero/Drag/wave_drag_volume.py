## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
# wave_drag_volume.py
# 
# Created:  Tim MacDonald, 6/24/14
# Modified: Tim MacDonald, 6/24/14
# 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import copy
import numpy as np

# ----------------------------------------------------------------------
#   Wave Drag Volume
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
def wave_drag_volume(conditions,configuration,wing):
    """Computes wave drag due to volume

    Assumptions:
    Simplified equations for wing

    Source:
    http://adg.stanford.edu/aa241/drag/ssdragcalc.html

    Inputs:
    conditions.freestream.mach_number        [Unitless]
    wing.thickness_to_chord                  [Unitless]
    wing.total_length                        [m]
    wing.areas.reference                     [m^2]

    Outputs:
    wave_drag_volume                         [Unitless]

    Properties Used:
    N/A
    """  

    # unpack inputs
    freestream   = conditions.freestream
    total_length = wing.total_length
    Sref         = wing.areas.reference
    
    # conditions
    Mc  = copy.copy(freestream.mach_number)
    
    # length-wise aspect ratio
    ARL = total_length**2/Sref
    
    # thickness to chord
    t_c_w = wing.thickness_to_chord
    
    # Computations
    x = np.pi*ARL/4
    beta = np.array([[0.0]] * len(Mc))
    wave_drag_volume = np.array([[0.0]] * len(Mc))    
    beta[Mc >= 1.05] = np.sqrt(Mc[Mc >= 1.05]**2-1)
    wave_drag_volume[Mc >= 1.05] = 4*t_c_w**2*(beta[Mc >= 1.05]**2+2*x**2)/(beta[Mc >= 1.05]**2+x**2)**1.5
    wave_drag_volume[0:len(Mc[Mc >= 1.05]),0] = wave_drag_volume[Mc >= 1.05]
    
    wave_drag_volume = wave_drag_volume * 1.15
    
    return wave_drag_volume