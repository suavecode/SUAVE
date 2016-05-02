# cl_r.py
# 
# Created:  Jun 2014, A. Wendorff
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------

def cl_r(cLw):
    """ output = SUAVE.Methods.Flight_Dynamics.Dynamic_Stablity.Full_Linearized_Equations.Supporting_Functions(CLw) 
        Calculating the rolling moment coefficient with respect to perturbational angular rate around the z-body-axis        
        Inputs:
                 
        Outputs:
                
        Assumptions:
        
        Source:
            J.H. Blakelock, "Automatic Control of Aircraft and Missiles" Wiley & Sons, Inc. New York, 1991, (Need page number)
    """

    # Generating Stability derivative
    cl_r = cLw/4.
    
    return cl_r