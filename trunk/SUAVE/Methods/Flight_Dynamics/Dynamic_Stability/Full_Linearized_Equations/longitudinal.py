## @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Full_Linearized_Equations
# longitudinal.py
# 
# Created:  Apr 2014, A. Wendorff
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
import numpy as np
import numpy.polynomial.polynomial as P

# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------

## @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Full_Linearized_Equations
def longitudinal(velocity, density, S_gross_w, mac, Cm_q, Cz_alpha, mass, Cm_alpha, Iy, Cm_alpha_dot, Cz_u, Cz_alpha_dot, Cz_q, Cw, Theta, Cx_u, Cx_alpha):
    """ This calculates the natural frequency and damping ratio for the full 
    linearized short period and phugoid modes        
    
    Assumptions:
        X-Z axis is plane of symmetry
        Constant mass of aircraft
        Origin of axis system at c.g. of aircraft
        Aircraft is a rigid body
        Earth is inertial reference frame
        Perturbations from equilibrium are small
        Flow is Quasisteady
        Zero initial conditions
        Cm_a = CF_z_a = CF_x_a = 0
        Neglect Cx_alpha_dot, Cx_q and Cm_u
        
    Source:
        J.H. Blakelock, "Automatic Control of Aircraft and Missiles" Wiley & Sons, Inc. New York, 1991, p 26-41.
        
    Inputs:
        velocity - flight velocity at the condition being considered                                          [meters/seconds]
        density - flight density at condition being considered                                                [kg/meters**3]
        S_gross_w - area of the wing                                                                          [meters**2]
        mac - mean aerodynamic chord of the wing                                                              [meters]
        Cm_q - coefficient for the change in pitching moment due to pitch rate                                [dimensionless] (2 * K * dC_m/di * lt/c where K is approximately 1.1)
        Cz_alpha - coefficient for the change in Z force due to the angle of attack                           [dimensionless] (-C_D - dC_L/dalpha)
        mass - mass of the aircraft                                                                           [kilograms]
        Cm_alpha - coefficient for the change in pitching moment due to angle of attack                       [dimensionless] (dC_m/dC_L * dCL/dalpha)
        Iy - moment of interia about the body y axis                                                          [kg * meters**2]
        Cm_alpha_dot - coefficient for the change in pitching moment due to rate of change of angle of attack [dimensionless] (2 * dC_m/di * depsilon/dalpha * lt/mac)
        Cz_u - coefficient for the change in force in the Z direction due to change in forward velocity       [dimensionless] (usually -2 C_L or -2C_L - U dC_L/du)
        Cz_alpha_dot - coefficient for the change of angle of attack caused by w_dot on the Z force           [dimensionless] (2 * dC_m/di * depsilon/dalpha)
        Cz_q - coefficient for the change in Z force due to pitching velocity                                 [dimensionless] (2 * K * dC_m/di where K is approximately 1.1)
        Cw - coefficient to account for gravity                                                               [dimensionless] (-C_L)
        Theta - angle between the horizontal axis and the body axis measured in the vertical plane            [radians]
        Cx_u - coefficient for the change in force in the X direction due to change in the forward velocity   [dimensionless] (-2C_D)
        Cx_alpha - coefficient for the change in force in the X direction due to the change in angle of attack caused by w [dimensionless] (C_L-dC_L/dalpha)
    
    Outputs:
        output - a data dictionary with fields:
            short_w_n - natural frequency of the short period mode                                            [radian/second]
            short_zeta - damping ratio of the short period mode                                               [dimensionless]
            phugoid_w_n - natural frequency of the short period mode                                          [radian/second]
            phugoid_zeta - damping ratio of the short period mode                                             [dimensionless]
        
    Properties Used:
    N/A  
    """ 

    # constructing matrix of coefficients
    A = (- Cx_u, mass * velocity / S_gross_w / (0.5*density*velocity**2.))  # X force U term
    B = (-Cx_alpha) # X force alpha term
    C = (-Cw * np.cos(Theta)) # X force theta term
    D = (-Cz_u) # Z force U term
    E = (-Cz_alpha, (mass*velocity/S_gross_w/(0.5*density*velocity**2.)-mac*0.5/velocity*Cz_alpha_dot)) # Z force alpha term
    F = (- Cw * np.sin(Theta), (-mass*velocity/S_gross_w/(0.5*density*velocity**2.)-mac*0.5/velocity*Cz_q))# Z force theta term
    G = (0.) # M moment U term
    H = (-Cm_alpha, -mac*0.5/velocity*Cm_alpha_dot) # M moment alpha term
    I = (0., - mac*0.5/velocity*Cm_q, Iy/S_gross_w/(0.5*density*velocity**2)/mac) # M moment theta term
    
    # Taking the determinant of the matrix ([A, B, C],[D, E, F],[G, H, I])
    EI    = P.polymul(E,I)
    FH    = P.polymul(F,H)
    part1 = P.polymul(A,P.polysub(EI,FH))
    DI    = P.polymul(D,I)
    FG    = P.polymul(F,G)
    part2 = P.polymul(B,P.polysub(FG,DI))    
    DH    = P.polymul(D,H)
    GE    = P.polymul(G,E)
    part3 = P.polymul(C,P.polysub(DH,GE))
    total = P.polyadd(part1,P.polyadd(part2,part3))
    
    # Use Synthetic division to split polynomial into two quadratic factors
    poly = total / total[4]
    poly1 = poly * 1. 
    poly2 = poly * 1. 
    poly3 = poly * 1.
    poly4 = poly * 1.
    
    poly1[4] = poly[4] - poly[2]/poly[2]
    poly1[3] = poly[3] - poly[1]/poly[2]
    poly1[2] = poly[2] - poly[0]/poly[2]
    poly2[4] = 0
    poly2[3] = poly1[3] - poly[2]*poly1[3]/poly[2]
    poly2[2] = poly1[2] - poly[1]*poly1[3]/poly[2]
    poly2[1] = poly1[1] - poly[0]*poly1[3]/poly[2]
    
    poly3[4] = poly[4] - poly2[2]/poly2[2]
    poly3[3] = poly[3] - poly2[1]/poly2[2]
    poly3[2] = poly[2] - poly2[0]/poly2[2]
    poly4[3] = poly3[3] - poly2[2]*poly3[3]/poly2[2]
    poly4[2] = poly3[2] - poly2[1]*poly3[3]/poly2[2]
    poly4[1] = poly3[1] - poly2[0]*poly3[3]/poly2[2]
     
    # Generate natural frequency and damping for Short Period and Phugoid modes
    short_w_n    = (poly4[2])**0.5
    short_zeta   = poly3[3]*0.5/short_w_n
    phugoid_w_n  = (poly2[0]/poly2[2])**0.5
    phugoid_zeta = poly2[1]/poly2[2]*0.5/phugoid_w_n
    
    output = Data()
    output.short_natural_frequency   = short_w_n
    output.short_damping_ratio       = short_zeta
    output.phugoid_natural_frequency = phugoid_w_n
    output.phugoid_damping_ratio     = phugoid_zeta    
    
    return output
