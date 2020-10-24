## @ingroupMethods-Noise-Fidelity_One-Engine
# secondary_noise_component.py
# 
# Created:  Jul 2015, C. Ilario
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------        
#   Imports
# ---------------------------------------------------------------------- 

import numpy as np

# ----------------------------------------------------------------------        
#   Secondary Noise Component
# ---------------------------------------------------------------------- 

## @ingroupMethods-Noise-Fidelity_One-Engine
def secondary_noise_component(SPL_s,Velocity_primary,theta_s,sound_ambient,Velocity_secondary,Velocity_aircraft,
                              Area_primary,Area_secondary,DSPL_s,EX_s,Str_s):
    """This function calculates the noise contribution of the secondary jet component
    
    Assumptions:
        None

    Source:
        None

    Inputs:
        SPL_s               [dB]
        Velocity_primary    [m/s]
        theta_s             [rad]
        sound_ambient       [dB]
        Velocity_secondary  [m/s]
        Velocity_aircraft   [m/s]
        Area_primary        [m^2]
        Area_secondary      [m^2]
        DSPL_s              [dB]
        EX_s                [-]
        Str_s               [-]

    Outputs: 
        SPL_s               [dB]

    Properties Used:
        N/A 
    
    """

    # Calculation of the velocity exponent
    velocity_exponent = 0.5 * 0.1*theta_s

    # Calculation of the Source Strengh Function (FV)
    FV = ((Velocity_secondary-Velocity_aircraft)/sound_ambient)**velocity_exponent * \
        ((Velocity_secondary+Velocity_aircraft)/sound_ambient)**(1-velocity_exponent)

    # Determination of the noise model coefficients
    Z1 = -18*((1.8*theta_s/np.pi)-0.6)**2
    Z2 = -14-8*((1.8*theta_s/np.pi)-0.6)**3
    Z3 = -0.7
    Z4 = 0.6 - 0.5*((1.8*theta_s/np.pi)-0.6)**2+0.5*(0.6-np.log10(1+Area_secondary/Area_primary))
    Z5 = 51 + 54*theta_s/np.pi - 9*((1.8*theta_s/np.pi)-0.6)**3
    Z6 = 99 + 36*theta_s/np.pi - 15*((1.8*theta_s/np.pi)-0.6)**4 + 5*Velocity_secondary*(Velocity_primary-Velocity_secondary)/(sound_ambient**2) + \
        DSPL_s + EX_s

    # Determination of Sound Pressure Level for the secondary jet component
    SPL_s = (Z1*np.log10(FV)+Z2)*(np.log10(Str_s)-Z3*np.log10(FV)-Z4)**2 + Z5*np.log10(FV) + Z6

    return SPL_s 

