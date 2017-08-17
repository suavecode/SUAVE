## @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Approximations
# phugoid.py
# 
# Created:  Apr 2014, A. Wendorff
# Modified: Jan 2015, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data

# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------

## @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Approximations
def phugoid(g, velocity, CD, CL):
    """ This calculates the natural frequency and damping ratio for the approximate 
    phugoid characteristics       

    Assumptions:
        constant angle of attack
        theta changes very slowly
        Inertial forces are neglected
        Neglect Cz_q
        Theta = 0
        X-Z axis is plane of symmetry
        Constant mass of aircraft
        Origin of axis system at c.g. of aircraft
        Aircraft is a rigid body
        Earth is inertial reference frame
        Perturbations from equilibrium are small
        Flow is Quasisteady 
        
    Source:
        J.H. Blakelock, "Automatic Control of Aircraft and Missiles" Wiley & Sons, Inc. New York, 1991, p 50-53.
        
    Inputs:
        g - gravitational constant                                   [meters/second**2]
        velocity - flight velocity at the condition being considered [meters/seconds]
        CD - coefficient of drag                                     [dimensionless]
        CL - coefficient of lift                                     [dimensionless]

    Outputs:
        output - a data dictionary with fields:
            phugoid_w_n - natural frequency of the phugoid mode      [radian/second]
            phugoid_zeta - damping ratio of the phugoid mode         [dimensionless]
                   
    Properties Used:
        N/A  
    """ 
    
    #process
    w_n = g/velocity * (2.)**0.5
    zeta = CD/(CL*(2.)**0.5)
    
    output = Data()
    output.natural_frequency = w_n
    output.damping_ratio = zeta
    
    return output