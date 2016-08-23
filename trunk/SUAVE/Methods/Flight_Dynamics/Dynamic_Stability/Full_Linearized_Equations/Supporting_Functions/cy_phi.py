# cy_phi.py
# 
# Created:  Jun 2014, A. Wendorff
# Modified: Jan 2016, E. Botero
# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------

def cy_phi(CL):
    """ output = SUAVE.Methods.Flight_Dynamics.Dynamic_Stablity.Full_Linearized_Equations.Supporting_Functions.cy_phi(CL) 
        Calculating the force coefficient in the y direction with respect to the roll angle of the aircraft        
        Inputs:
                 
        Outputs:
                
        Assumptions:
        
        Source:
            J.H. Blakelock, "Automatic Control of Aircraft and Missiles" Wiley & Sons, Inc. New York, 1991, (Need page number)
    """

    # Generating Stability derivative
    
    cy_phi = CL
    
    return cy_phi