## @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Full_Linearized_Equations-Supporting_Functions
# cn_p.py
# 
# Created:  Jun 2014, A. Wendorff
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------
## @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Full_Linearized_Equations-Supporting_Functions
def cn_p(cLw,depdalpha):
    """ This calculats the yawing moment coefficient with respect
    to perturbational angular rate around the x-body-axis            

    Assumptions:
    None
    
    Source:
    J.H. Blakelock, "Automatic Control of Aircraft and Missiles" 
    Wiley & Sons, Inc. New York, 1991, (pg 23)
    
    Inputs:
    clw          [dimensionless]
    dep_alpha    [dimensionless]
             
    Outputs:
    cn_p         [dimensionless]
            
    Properties Used:
    N/A           
    """

    # Generating Stability derivative
    cn_p = -cLw/8. * (1. - depdalpha)
    
    return cn_p