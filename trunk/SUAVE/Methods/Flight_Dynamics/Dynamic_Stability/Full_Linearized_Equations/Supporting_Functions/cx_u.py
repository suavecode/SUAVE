## @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Full_Linearized_Equations-Supporting_Functions
# cx_u.py
# 
# Created:  Jun 2014, A. Wendorff
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------

## @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Full_Linearized_Equations-Supporting_Functions
def cx_u(cD):
    """ This calculates the coefficient of force in the x direction
    with respect to the change in forward velocity of the aircraft        

    Assumptions:
    None

    Source:
    J.H. Blakelock, "Automatic Control of Aircraft and Missiles" 
    Wiley & Sons, Inc. New York, 1991, (pg 23)

    Inputs:
    cD                  [dimensionless]

    Outputs:
    cx_u                [dimensionless]

    Properties Used:
    N/A           
    """

    # Generating Stability derivative
    cx_u  = -2. * cD
    
    return cx_u