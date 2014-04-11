# test_dynamicstability.py

import SUAVE
import numpy as np
from SUAVE.Attributes import Units as Units
from SUAVE.Methods.Flight_Dynamics.Dynamic_Stability import Approximations as Approximations
from SUAVE.Structure import (
    Data, Container, Data_Exception, Data_Warning,
)

# Taken from Blakelock

#Lateral/Directional Inputs
velocity = 440 * (Units.ft/Units.sec) # Flight Velocity
Cn_Beta = 0.096 # Yawing moment coefficient due to sideslip
S_gross_w = 2400 * (Units.ft**2) # Wing reference area
density = 0.002378 * (Units.slugs/Units.ft**3) # Sea level density
span = 130 * Units.ft # Span of the aircraft
mass = 5900 * Units.slugs # mass of the aircraft
I_x = 1.995 * 10**6 * (Units.slugs*Units.ft**2) # Moment of Inertia in x-axis
I_z = 4.2 * 10**6 * (Units.slugs*Units.ft**2) # Moment of Inertia in z-axis
Cn_r = -0.107 # Yawing moment coefficient due to yawing velocity
Cl_p = -0.38 #  Rolling moment coefficient due to the rolling velocity
Cy_phi = 0.344 # Side force coefficient due to aircraft roll
Cl_Beta = -0.057 # Rolling moment coefficient due to to sideslip
Cl_r = 0.086 # Rolling moment coefficient due to the yawing velocity (usually evaluated analytically)
    
dutch_roll=Approximations.dutch_roll(velocity, Cn_Beta, S_gross_w, density, span, I_z, Cn_r)
roll_tau = Approximations.roll(I_x, S_gross_w, density, velocity, span, Cl_p)
spiral_tau = Approximations.spiral(mass, velocity, density, S_gross_w, Cl_p, Cn_Beta, Cy_phi, Cl_Beta, Cn_r, Cl_r)    
    
# Longitudinal Inputs
mass= 5800 * Units.slugs # mass of the aircraft
velocity = 600 * (Units.ft/Units.sec) # Flight Velocity
mac = 20.2 * Units.ft # mean aerodynamic chord
Cm_q = -11.4 # Change in pitching moment coefficient due to pitching velocity
CL = 0.74 # Coefficient of Lift
CD = 0.04 # Coefficient of Drag
CL_alpha = 4.42 # Change in aircraft lift due to change in angle of attack
Cz_alpha = -CL_alpha - CD
SM = -0.14 # static margin
Cm_alpha = SM * CL_alpha
Cm_alpha_dot = -3.27
Iy = 2.62 * 10**6 * (Units.slugs*Units.ft**2) # Moment of Inertia in y-axis
density = 0.000585 * (Units.slugs/Units.ft**3) # 40,000 ft density
g = 9.8 # gravitational constant

short_period = Approximations.short_period(velocity, density, S_gross_w, mac, Cm_q, Cz_alpha, mass, Cm_alpha, Iy, Cm_alpha_dot)
phugoid = Approximations.phugoid(g, velocity, CD, CL)

print dutch_roll