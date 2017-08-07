## @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Full_Linearized_Equations-Supporting_Functions
# cm_q.py
# 
# Created:  Jun 2014, A. Wendorff
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------

## @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Full_Linearized_Equations-Supporting_Functions
def cm_q(cm_i, l_t, mac):
    """ This calculates the damping in pitch coefficient        

    Assumptions:
    None

    Source:
    J.H. Blakelock, "Automatic Control of Aircraft and Missiles" 
    Wiley & Sons, Inc. New York, 1991, (pg 23)

    Inputs:
    cm_i                [dimensionless]
    l_t                 [meters]
    mac                 [meters]

    Outputs:
    cm_q                [dimensionless]

    Properties Used:
    N/A           
    """
    cm_q = 2. * 1.1 * cm_i * l_t / mac 
    
    return cm_q