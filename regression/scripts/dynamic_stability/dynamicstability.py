# test_dynamicstability.py

import SUAVE
import numpy as np
from SUAVE.Core import Units
from SUAVE.Methods.Flight_Dynamics.Dynamic_Stability import Approximations as Approximations
from SUAVE.Methods.Flight_Dynamics.Dynamic_Stability import Full_Linearized_Equations as Full_Linearized_Equations
from SUAVE.Core import (
    Data, Container,
)

def main():

    # Taken from Blakelock
    
    #Lateral/Directional Inputs
    velocity = 440 * (Units['ft/sec']) # Flight Velocity
    Cn_Beta = 0.096 # Yawing moment coefficient due to sideslip
    S_gross_w = 2400 * (Units['ft**2']) # Wing reference area
    density = 0.002378 * (Units['slugs/ft**3']) # Sea level density
    span = 130 * Units.ft # Span of the aircraft
    mass = 5900 * Units.slugs # mass of the aircraft
    I_x = 1.995 * 10**6 * (Units['slugs*ft**2']) # Moment of Inertia in x-axis
    I_z = 4.2 * 10**6 * (Units['slugs*ft**2']) # Moment of Inertia in z-axis
    Cn_r = -0.107 # Yawing moment coefficient due to yawing velocity
    Cl_p = -0.38 #  Rolling moment coefficient due to the rolling velocity
    Cy_phi = 0.344 # Side force coefficient due to aircraft roll
    Cl_Beta = -0.057 # Rolling moment coefficient due to to sideslip
    Cl_r = 0.086 # Rolling moment coefficient due to the yawing velocity (usually evaluated analytically)
    J_xz = 0 # Assumed
    Cn_p = -0.0228
    Cy_psi = 0
    Cy_Beta = -0.6
    mass = 5900 * Units.slugs # mass of the aircraft
    
       
    dutch_roll=Approximations.dutch_roll(velocity, Cn_Beta, S_gross_w, density, span, I_z, Cn_r)
    roll_tau = Approximations.roll(I_x, S_gross_w, density, velocity, span, Cl_p)
    spiral_tau = Approximations.spiral(mass, velocity, density, S_gross_w, Cl_p, Cn_Beta, Cy_phi, Cl_Beta, Cn_r, Cl_r)
    lateral_directional = Full_Linearized_Equations.lateral_directional(velocity, Cn_Beta, S_gross_w, density, span, I_z, Cn_r, I_x, Cl_p, J_xz, Cl_r, Cl_Beta, Cn_p, Cy_phi, Cy_psi, Cy_Beta, mass)
        
    # Longitudinal Inputs
    mass= 5800 * Units.slugs # mass of the aircraft
    velocity = 600 * (Units['ft/sec']) # Flight Velocity
    mac = 20.2 * Units.ft # mean aerodynamic chord
    Cm_q = -11.4 # Change in pitching moment coefficient due to pitching velocity
    CL = 0.74 # Coefficient of Lift
    CD = 0.044 # Coefficient of Drag
    CL_alpha = 4.42 # Change in aircraft lift due to change in angle of attack
    Cz_alpha = -4.46
    SM = -0.14 # static margin
    Cm_alpha = SM * CL_alpha
    Cm_alpha_dot = -3.27
    Iy = 2.62 * 10**6 * (Units['slugs*ft**2']) # Moment of Inertia in y-axis
    density = 0.000585 * (Units['slugs/ft**3']) # 40,000 ft density
    g = 9.8 # gravitational constant
    Cz_u = -2*CL # change in Z force with respect to change in forward velocity
    Cm_alpha_dot = -3.27
    Cz_q = -3.94
    Cw = -CL 
    Theta = 0
    Cx_u = -2*CD
    Cx_alpha = 0.392
    Cz_alpha_dot = -1.13
    
    short_period = Approximations.short_period(velocity, density, S_gross_w, mac, Cm_q, Cz_alpha, mass, Cm_alpha, Iy, Cm_alpha_dot)
    phugoid = Approximations.phugoid(g, velocity, CD, CL)
    longitudinal = Full_Linearized_Equations.longitudinal(velocity, density, S_gross_w, mac, Cm_q, Cz_alpha, mass, Cm_alpha, Iy, Cm_alpha_dot, Cz_u, Cz_alpha_dot, Cz_q, Cw, Theta, Cx_u, Cx_alpha)
    
    # Expected Values
    Blakelock = Data()
    Blakelock.longitudinal_short_zeta = 0.352
    Blakelock.longitudinal_short_w_n = 1.145
    Blakelock.longitudinal_phugoid_zeta = 0.032
    Blakelock.longitudinal_phugoid_w_n = 0.073
    Blakelock.short_period_short_w_n = 1.15
    Blakelock.short_period_short_zeta = 0.35
    Blakelock.phugoid_phugoid_w_n = 0.0765
    Blakelock.phugoid_phugoid_zeta = 0.042
    Blakelock.lateral_directional_dutch_w_n = 1.345
    Blakelock.lateral_directional_dutch_zeta = 0.14
    Blakelock.lateral_directional_spiral_tau = 1/0.004
    Blakelock.lateral_directional_roll_tau = -1/2.09
    Blakelock.dutch_roll_dutch_w_n = 1.28
    Blakelock.dutch_roll_dutch_zeta = 0.114
    Blakelock.spiral_tau = 1./0.0042
    Blakelock.roll_tau = -0.493
    
    # Calculating error percentage
    error = Data()
    error.longitudinal_short_zeta = (Blakelock.longitudinal_short_zeta - longitudinal.short_damping_ratio) / Blakelock.longitudinal_short_zeta
    error.longitudinal_short_w_n = (Blakelock.longitudinal_short_w_n - longitudinal.short_natural_frequency) / Blakelock.longitudinal_short_w_n
    error.longitudinal_phugoid_zeta = (Blakelock.longitudinal_phugoid_zeta - longitudinal.phugoid_damping_ratio) / Blakelock.longitudinal_phugoid_zeta
    error.longitudinal_phugoid_w_n = (Blakelock.longitudinal_phugoid_w_n - longitudinal.phugoid_natural_frequency) / Blakelock.longitudinal_phugoid_w_n
    error.short_period_w_n = (Blakelock.short_period_short_w_n - short_period.natural_frequency) / Blakelock.short_period_short_w_n
    error.short_period_short_zeta = (Blakelock.short_period_short_zeta - short_period.damping_ratio) / Blakelock.short_period_short_zeta
    error.phugoid_phugoid_w_n = (Blakelock.phugoid_phugoid_w_n - phugoid.natural_frequency) / Blakelock.phugoid_phugoid_w_n
    error.phugoid_phugoid_zeta = (Blakelock.phugoid_phugoid_zeta - phugoid.damping_ratio) / Blakelock.phugoid_phugoid_zeta
    error.lateral_directional_dutch_w_n = (Blakelock.lateral_directional_dutch_w_n - lateral_directional.dutch_natural_frequency) / Blakelock.lateral_directional_dutch_w_n
    error.lateral_directional_dutch_zeta = (Blakelock.lateral_directional_dutch_zeta - lateral_directional.dutch_damping_ratio) / Blakelock.lateral_directional_dutch_zeta                              
    error.lateral_directional_spiral_tau = (Blakelock.lateral_directional_spiral_tau - lateral_directional.spiral_tau) / Blakelock.lateral_directional_spiral_tau
    error.lateral_directional_roll_tau = (Blakelock.lateral_directional_roll_tau - lateral_directional.roll_tau) / Blakelock.lateral_directional_roll_tau
    error.dutch_roll_dutch_w_n = (Blakelock.dutch_roll_dutch_w_n - dutch_roll.natural_frequency) / Blakelock.dutch_roll_dutch_w_n
    error.dutch_roll_dutch_zeta = (Blakelock.dutch_roll_dutch_zeta - dutch_roll.damping_ratio) / Blakelock.dutch_roll_dutch_zeta
    error.spiral_tau = (Blakelock.spiral_tau - spiral_tau) / Blakelock.spiral_tau                               
    error.roll_tau = (Blakelock.roll_tau - roll_tau) / Blakelock.roll_tau
    
    print(error)
    
    for k,v in list(error.items()):
        assert(np.abs(v)<0.08)
        
        
    return

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
    
