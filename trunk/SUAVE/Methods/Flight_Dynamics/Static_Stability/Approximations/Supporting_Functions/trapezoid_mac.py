## @ingroup Methods-Flight_Dynamics-Static_Stability-Approximations-Supporting_Functions
# trapezoid_mac.py
#
# Created:  Mar 2014, T. Momose
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Method
# ----------------------------------------------------------------------

## @ingroup Methods-Flight_Dynamics-Static_Stability-Approximations-Supporting_Functions
def trapezoid_mac(wing):
    """ This method computes the mean aerodynamic chord of a linearly tapered
    trapezoidal aerodynamic surface

    Assumptions:
        Assumes a simple trapezoidal wing shape.
        
    Source:
        Unknown

    Inputs:
        wing - a data dictionary with the fields:
            areas.reference - the planform area of the trapezoidal wing               [meters**2]
            spans.projected - wing span                                               [meters]
            chords.root - the wing root chord                                         [meters]
            {One of the following}
            chords.tip - the wing tip chord                                           [meters]
            taper - taper ratio of the wing                                           [dimensionless]

    Outputs:
        mac - the mean aerodynamic chord of the wing (or equivalent trapezoid)        [meters]

    Properties Used:
        N/A
    """                 

    #Unpack inputs
    S   = wing.areas.reference
    b   = wing.spans.projected
    l   = wing.taper
    c_r = wing.chords.root
    c_t = wing.chords.tip
    mac = wing.chords.mean_aerodynamic
    
    #Get MAC
    if mac:
        return mac    
    
    # Compute root and tip chords. Find root and tip chords from wing area, span,
    # and taper if the user has not specified the root and tip chords
    if not c_t:
        if not c_r:
            mgc = S/b               #mean geometric chord
            c_r = mgc/(1-0.5*(1-l)) #root chord
            
        c_t = c_r*l                 #tip chord
    
    #Compute mean aerodynamic chord
    mac = (2.0/3.0)*(c_r + c_t - c_r*c_t/(c_r + c_t))
    
    return mac
