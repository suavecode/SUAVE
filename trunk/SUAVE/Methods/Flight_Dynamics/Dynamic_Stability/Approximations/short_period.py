## @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Approximations
# short_period.py
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
def short_period(velocity, density, S_gross_w, mac, Cm_q, Cz_alpha, mass, Cm_alpha, Iy, Cm_alpha_dot):
    """ This calculates the natural frequency and damping ratio for the approximate short
    period characteristics        
            
    Assumptions:
        X-Z axis is plane of symmetry
        Constant mass of aircraft
        Origin of axis system at c.g. of aircraft
        Aircraft is a rigid body
        Earth is inertial reference frame
        Perturbations from equilibrium are small
        Flow is Quasisteady
        Constant forward airspeed
        Neglect Cz_alpha_dot and Cz_q
        Theta = 0
        
    Source:
        J.H. Blakelock, "Automatic Control of Aircraft and Missiles" Wiley & Sons, Inc. New York, 1991, p 46-50.
        
    Inputs:
        velocity - flight velocity at the condition being considered                                          [meters/seconds]
        density - flight density at condition being considered                                                [kg/meters**3]
        S_gross_w - area of the wing                                                                          [meters**2]
        mac - mean aerodynamic chord of the wing                                                              [meters]
        Cm_q - coefficient for the change in pitching moment due to pitch rate                                [dimensionless]
        Cz_alpha - coefficient for the change in Z force due to the angle of attack                           [dimensionless]
        mass - mass of the aircraft                                                                           [kilograms]
        Cm_alpha - coefficient for the change in pitching moment due to angle of attack                       [dimensionless]
        Iy - moment of interia about the body y axis                                                          [kg * meters**2]
        Cm_alpha_dot - coefficient for the change in pitching moment due to rate of change of angle of attack [dimensionless]
    
    Outputs:
        output - a data dictionary with fields:
        w_n - natural frequency of the short period mode                                                      [radian/second]
        zeta - damping ratio of the short period mode                                                         [dimensionless]
    
    Properties Used:
        N/A          
    """ 
    
    #process
    w_n = velocity * density * S_gross_w * mac / 2. * ((0.5*Cm_q*Cz_alpha - 2. * mass / density /S_gross_w /mac * Cm_alpha) / Iy /mass)**(0.5)
    zeta = -0.25 * (Cm_q + Cm_alpha_dot + 2. * Iy * Cz_alpha / mass / (mac ** 2.)) * ( mass * mac ** 2. / Iy / (Cm_q * Cz_alpha * 0.5 - 2. * mass * Cm_alpha / density / S_gross_w / mac)) ** 0.5
    
    output = Data()
    output.natural_frequency = w_n
    output.damping_ratio = zeta
    
    return output