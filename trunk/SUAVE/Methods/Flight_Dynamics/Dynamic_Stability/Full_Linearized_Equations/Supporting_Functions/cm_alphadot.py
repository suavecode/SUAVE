# cm_alphadot.py
# 
# Created:  Jun 2014, A. Wendorff
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------

def cm_alphadot(cm_i, ep_alpha, l_t, mac):
    """ output = SUAVE.Methods.Flight_Dynamics.Dynamic_Stablity.Full_Linearized_Equations.Supporting_Functions.cm_alphadot(cm_i, ep_alpha, l_t, mac) 
        Calculating the pitching moment coefficient with respect to the rate of change of the alpha of attack of the aircraft        
        Inputs:
                 
        Outputs:
                
        Assumptions:
        
        Source:
            J.H. Blakelock, "Automatic Control of Aircraft and Missiles" Wiley & Sons, Inc. New York, 1991, (Need page number)
    """

    # Generating Stability derivative

    cm_alphadot  = 2. * cm_i * ep_alpha * l_t / mac
    
    return cm_alphadot 