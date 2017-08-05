## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
# wave_drag_body_of_rev.py
# 
# Created:  Jun 2014, T. Macdonald
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
#   wave drag body of rev
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
def wave_drag_body_of_rev(total_length,Rmax,Sref):
    """Use wave drag to determine compressibility drag a body of revolution

    Assumptions:
    Corrected Sear-Haack body 

    Source:
    adg.stanford.edu (Stanford AA241 A/B Course Notes)

    Inputs:
    total_length                    [m]
    Rmax (max radius)               [m]
    Sref (main wing reference area) [m^2]

    Outputs:
    wave_drag_body_of_rev*1.15      [Unitless]

    Properties Used:
    N/A
    """  
    # Computations - takes drag of Sears-Haack and use wing reference area for CD
    wave_drag_body_of_rev = (9.0*(np.pi)**3.0*Rmax**4.0/(4.0*total_length**2.0))/(0.5*Sref)
    
    return wave_drag_body_of_rev
