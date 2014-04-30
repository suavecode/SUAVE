# phugoid.py
# 
# Created:  Andrew Wendorff, April 2014
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import (
    Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------

def phugoid(g, velocity, CD, CL):
    """ output = SUAVE.Methods.Flight_Dynamics.Dynamic_Stablity.Approximations.phugoid(g, velocity, CD, CL)
        Calculate the natural frequency and damping ratio for the approximate phugoid characteristics       
        
        Inputs:
            g - gravitational constant [meters/second**2]
            velocity - flight velocity at the condition being considered [meters/seconds]
            CD - coefficient of drag [dimensionless]
            CL - coefficient of lift [dimensionless]

        Outputs:
            output - a data dictionary with fields:
                phugoid_w_n - natural frequency of the phugoid mode [radian/second]
                phugoid_zeta - damping ratio of the phugoid mode [dimensionless]
            
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
    """ 
    
    #process
    w_n = g/velocity * (2.)**0.5
    zeta = CD/(CL*(2.)**0.5)
    
    output = Data()
    output.phugoid_w_n = w_n
    output.phugoid_zeta = zeta
    
    return output