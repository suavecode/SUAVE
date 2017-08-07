## @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Approximations
# roll.py
# 
# Created:  Apr 2014, A. Wendorff
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data

# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------

## @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Approximations
def roll(I_x, S_gross_w, density, velocity, span, Cl_p):
    """ This calculates the approximate time constant for the roll mode       

    Assumptions:
       Only the rolling moment equation is needed from the Lateral-Directional equations
       Sideslip and yaw angle are being neglected and thus set to be zero.
       delta_r = 0
       X-Z axis is plane of symmetry
       Constant mass of aircraft
       Origin of axis system at c.g. of aircraft
       Aircraft is a rigid body
       Earth is inertial reference frame
       Perturbations from equilibrium are small
       Flow is Quasisteady
       
    Source:
       J.H. Blakelock, "Automatic Control of Aircraft and Missiles" Wiley & Sons, Inc. New York, 1991, p 134-135.
       
    Inputs:
       I_x -  moment of interia about the body x axis [kg * meters**2]
       S_gross_w - area of the wing [meters**2]
       density - flight density at condition being considered [kg/meters**3]
       span - wing span of the aircraft [meters]
       velocity - flight velocity at the condition being considered [meters/seconds]
       Cl_p - change in rolling moment due to the rolling velocity [dimensionless]
   
    Outputs:
       roll_tau - approximation of the time constant of the roll mode of an aircraft [seconds] (positive values are bad)

    Properties Used:
       N/A         
    """ 
    
    #process
    roll_tau = 4.*I_x/(S_gross_w*density*velocity*span**2.*Cl_p)
    
    return roll_tau