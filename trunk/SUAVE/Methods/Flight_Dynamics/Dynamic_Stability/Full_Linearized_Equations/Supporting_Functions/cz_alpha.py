# @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Full_Linearized_Equations-Supporting_Functions
# cz_alpha.py
# 
# Created:  Jun 2014, A. Wendorff
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------

# @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Full_Linearized_Equations-Supporting_Functions
def cz_alpha(cD, cL_alpha):
    """ This calculates the coefficient of force in the z-direction
    with respect to alpha of attack of the aircraft        

    Assumptions:
    None

    Source:
    J.H. Blakelock, "Automatic Control of Aircraft and Missiles" 
    Wiley & Sons, Inc. New York, 1991, (pg 23)

    Inputs:
    cD                         [dimensionless]
    cL_alpha                   [dimensionless]

    Outputs:
    cz_alpha                   [dimensionless]

    Properties Used:
    N/A           
    """

    # Generating Stability derivative

    cz_alpha  = -cD - cL_alpha
    
    return cz_alpha 