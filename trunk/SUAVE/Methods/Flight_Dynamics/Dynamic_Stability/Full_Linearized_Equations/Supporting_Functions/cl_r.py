## @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Full_Linearized_Equations-Supporting_Functions
# cl_r.py
# 
# Created:  Jun 2014, A. Wendorff
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------
## @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Full_Linearized_Equations-Supporting_Functions
def cl_r(cLw):
    """ This calculates the rolling moment coefficient with respect to
    perturbational angular rate around the z-body-axis        
    
    Assumptions:
    None
    
    Source:
    J.H. Blakelock, "Automatic Control of Aircraft and Missiles" 
    Wiley & Sons, Inc. New York, 1991, (pg 23
    
    Inputs:
    clw          [dimensionless]
             
    Outputs:
    cl_r         [dimensionless]
            
    Properties Used:
    N/A                
    """

    # Generating Stability derivative
    cl_r = cLw/4.
    
    return cl_r