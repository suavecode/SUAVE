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

def wave_drag_body_of_rev(total_length,Rmax,Sref):
    """ SUAVE.Methods.wave_drag_lift(conditions,configuration,fuselage)
        Based on http://adg.stanford.edu/aa241/drag/ssdragcalc.html
        computes the wave drag due to lift 
        
        Inputs:

        Outputs:

        Assumptions:

        
    """
    # Computations - takes drag of Sears-Haack and use wing reference area for CD
    wave_drag_body_of_rev = (9.0*(np.pi)**3.0*Rmax**4.0/(4.0*total_length**2.0))/(0.5*Sref)
    
    return wave_drag_body_of_rev
