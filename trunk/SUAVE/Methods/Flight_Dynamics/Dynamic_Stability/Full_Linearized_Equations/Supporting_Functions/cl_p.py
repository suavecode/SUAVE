## @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Full_Linearized_Equations-Supporting_Functions
# cl_p.py
#
# Created:  Aug 2016, A. van Korlaar
# Modified: Aug 2016, L. Kulik

# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------

## @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Full_Linearized_Equations-Supporting_Functions
def cl_p(cl_alpha, geometry):
    """ This calculates the derivative of rolling moment with respect 
    to roll rate

    Assumptions:
    None
    
    Source:
    STABILITY, USAF. "Control Datcom." Air Force Flight Dynamics 
    Laboratory, Wright-Patterson Air Force Base, Ohio (1972)
    
    Inputs:
    taper        [dimensionless]
    cl_alpha     [dimensionless]   

    Outputs:
    cl_p         [dimensionless]
    
    Properties Used:
    N/A
    """

    taper = geometry.wings['main_wing'].taper

    # Generating Stability derivative
    cl_p = -(cl_alpha/12.)*((1.+3.*taper)/(1.+taper))

    return cl_p