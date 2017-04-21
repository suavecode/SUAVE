# @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Full_Linearized_Equations-Supporting_Functions
# cx_u.py
# 
# Created:  Jun 2014, A. Wendorff
# Modified: Jan 2016, E. Botero


# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------

# @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Full_Linearized_Equations-Supporting_Functions
def cx_u(cD):
    """ output = SUAVE.Methods.Flight_Dynamics.Dynamic_Stablity.Full_Linearized_Equations.Supporting_Functions.cx_u(cD) 
        Calculating the coefficient of force in the x direction with respect to the change in forward velocity of the aircraft        
        Inputs:
                 
        Outputs:
                
        Assumptions:
        
        Source:
            J.H. Blakelock, "Automatic Control of Aircraft and Missiles" Wiley & Sons, Inc. New York, 1991, (Need page number)
    """

    # Generating Stability derivative
    cx_u  = -2. * cD
    
    return cx_u