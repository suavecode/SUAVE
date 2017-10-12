## @ingroup Methods-Flight_Dynamics-Dynamic_Stability-Full_Linearized_Equations
# lateral_directional.py
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
def lateral_directional(velocity, Cn_Beta, S_gross_w, density, span, I_z, Cn_r, I_x, Cl_p, J_xz, Cl_r, Cl_Beta, Cn_p, Cy_phi, Cy_psi, Cy_Beta, mass):
    """ This calculates the natural frequency and damping ratio for the full linearized dutch 
    roll mode along with the time constants for the roll and spiral modes   
    
    Assumptions:
        X-Z axis is plane of symmetry
        Constant mass of aircraft
        Origin of axis system at c.g. of aircraft
        Aircraft is a rigid body
        Earth is inertial reference frame
        Perturbations from equilibrium are small
        Flow is Quasisteady
        Zero initial conditions
        Neglect Cy_p and Cy_r
        
    Source:
        J.H. Blakelock, "Automatic Control of Aircraft and Missiles" Wiley & Sons, Inc. New York, 1991, p 118-124.
        
    Inputs:
        velocity - flight velocity at the condition being considered                          [meters/seconds]
        Cn_Beta - coefficient for change in yawing moment due to sideslip                     [dimensionless] (no simple relation)
        S_gross_w - area of the wing                                                          [meters**2]
        density - flight density at condition being considered                                [kg/meters**3]
        span - wing span of the aircraft                                                      [meters]
        I_z - moment of interia about the body z axis                                         [kg * meters**2]
        Cn_r - coefficient for change in yawing moment due to yawing velocity                 [dimensionless] ( - C_D(wing)/4 - 2 * Sv/S * (l_v/b)**2 * (dC_L/dalpha)(vert) * eta(vert))
        I_x - moment of interia about the body x axis                                         [kg * meters**2]
        Cl_p - change in rolling moment due to the rolling velocity                           [dimensionless] (no simple relation for calculation)
        J_xz - products of inertia in the x-z direction                                       [kg * meters**2] (if X and Z lie in a plane of symmetry then equal to zero)
        Cl_r - coefficient for change in rolling moment due to yawing velocity                [dimensionless] (Usually equals C_L(wing)/4)
        Cl_Beta - coefficient for change in rolling moment due to sideslip                    [dimensionless] 
        Cn_p - coefficient for the change in yawing moment due to rolling velocity            [dimensionless] (-C_L(wing)/8*(1 - depsilon/dalpha)) (depsilon/dalpha = 2/pi/e/AspectRatio dC_L(wing)/dalpha)
        Cy_phi  - coefficient for change in sideforce due to aircraft roll                    [dimensionless] (Usually equals C_L)
        Cy_psi - coefficient to account for gravity                                           [dimensionless] (C_L * tan(Theta))
        Cy_Beta - coefficient for change in Y force due to sideslip                           [dimensionless] (no simple relation)
        mass - mass of the aircraft                                                           [kilograms]
    
    Outputs:
        output - a data dictionary with fields:
        dutch_w_n - natural frequency of the dutch roll mode                                  [radian/second]
        dutch_zeta - damping ratio of the dutch roll mode                                     [dimensionless]
        roll_tau - approximation of the time constant of the roll mode of an aircraft         [seconds] (positive values are bad)
        spiral_tau - time constant for the spiral mode                                        [seconds] (positive values are bad)
    
    Properties Used:
        N/A         
    """ 
    
    # constructing matrix of coefficients
    A = (0, -span * 0.5 / velocity * Cl_p, I_x/S_gross_w/(0.5*density*velocity**2)/span )  # L moment phi term
    B = (0, -span * 0.5 / velocity * Cl_r, -J_xz / S_gross_w / (0.5 * density * velocity ** 2.) / span) # L moment psi term
    C = (-Cl_Beta) # L moment Beta term
    D = (0, - span * 0.5 / velocity * Cn_p, -J_xz / S_gross_w / (0.5 * density * velocity ** 2.) / span ) # N moment phi term 
    E = (0, - span * 0.5 / velocity * Cn_r, I_z / S_gross_w / (0.5 * density * velocity ** 2.) / span ) # N moment psi term
    F = (-Cn_Beta) # N moment Beta term
    G = (-Cy_phi) # Y force phi term
    H = (-Cy_psi, mass * velocity / S_gross_w / (0.5 * density * velocity ** 2.))    
    I = (-Cy_Beta, mass * velocity / S_gross_w / (0.5 * density * velocity ** 2.))
    
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
    poly  = total / total[5]
    
    # Generate the time constant for the spiral and roll modes along with the damping and natural frequency for the dutch roll mode
    root       = np.roots(poly)
    root       = sorted(root,reverse=True)
    spiral_tau = 1 * root[0].real
    w_n        = (root[1].imag**2 + root[1].real**2)**(-0.5)
    zeta       = -2*root[1].real/w_n
    roll_tau   = 1 * root [3].real
    
    output = Data()
    output.dutch_natural_frequency = w_n
    output.dutch_damping_ratio     = zeta
    output.spiral_tau              = spiral_tau
    output.roll_tau                = roll_tau    
    
    return output