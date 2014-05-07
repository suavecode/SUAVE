# dutch_roll.py
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

def dutch_roll(velocity, Cn_Beta, S_gross_w, density, span, I_z, Cn_r):
    """ output = SUAVE.Methods.Flight_Dynamics.Dynamic_Stablity.Approximations.dutch_roll(velocity, Cn_Beta, S_gross_w, density, span, I_z, Cn_r)
        Calculate the natural frequency and damping ratio for the approximate dutch roll characteristics       
        
        Inputs:
            velocity - flight velocity at the condition being considered [meters/seconds]
            Cn_Beta - coefficient for change in yawing moment due to sideslip [dimensionless]
            S_gross_w - area of the wing [meters**2]
            density - flight density at condition being considered [kg/meters**3]
            span - wing span of the aircraft [meters]
            I_z - moment of interia about the body z axis [kg * meters**2]
            Cn_r - coefficient for change in yawing moment due to yawing velocity [dimensionless]
        
        Outputs:
            output - a data dictionary with fields:
                dutch_w_n - natural frequency of the dutch roll mode [radian/second]
                dutch_zeta - damping ratio of the dutch roll mode [dimensionless]
            
        Assumptions:
            Major effect of rudder deflection is the generation of the Dutch roll mode.
            Dutch roll mode only consists of sideslip and yaw
            Beta = -Psi
            Phi and its derivatives are zero
            consider only delta_r input and Theta = 0
            Neglect Cy_r
            X-Z axis is plane of symmetry
            Constant mass of aircraft
            Origin of axis system at c.g. of aircraft
            Aircraft is a rigid body
            Earth is inertial reference frame
            Perturbations from equilibrium are small
            Flow is Quasisteady
            
        Source:
            J.H. Blakelock, "Automatic Control of Aircraft and Missiles" Wiley & Sons, Inc. New York, 1991, p 132-134.
    """ 
    
    #process
    w_n = velocity * (Cn_Beta*S_gross_w*density*span/2./I_z)**0.5 # natural frequency
    zeta = -Cn_r /8. * (2.*S_gross_w*density*span**3./I_z/Cn_Beta)**0.5 # damping ratio
    
    output = Data() 
    output.dutch_w_n = w_n
    output.dutch_zeta = zeta
    
    
    return output