## @ingroupMethods-Noise-Fidelity_One-Engine
# mixed_noise_component.py
# 
# Created:  Jul 2015, C. Ilario
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------        
#   Imports
# ---------------------------------------------------------------------- 

import numpy as np

# ----------------------------------------------------------------------        
#   Mixed Noise Component
# ---------------------------------------------------------------------- 

## @ingroupMethods-Noise-Fidelity_One-Engine
def mixed_noise_component(SPL_m,Velocity_primary,theta_m,sound_ambient,Velocity_secondary,
                          Velocity_aircraft,Area_primary,Area_secondary,DSPL_m,EX_m,Str_m,Velocity_mixed,XBPR):
    """This function calculates the noise contribution of the mixed jet component
    
    Assumptions:
        N/A

    Source:
        None

    Inputs:
        SPL_m               [dB]
        Velocity_primary    [m/s]
        theta_m             [rad]
        sound_ambient       [SPL]
        Velocity_secondary  [m/s]
        Velocity_aircraft   [m/s]
        Area_primary        [m^2]
        Area_secondary      [m^2]
        DSPL_m              [SPL]
        EX_m
        Str_m
        Velocity_mixed      [m/s]
        XBPR                   

    Outputs:
        SPL_m                [dB]

    Properties Used: 
        N/A 
    """ 

    #Calculation of the velocity exponent
    velocity_exponent = (Velocity_mixed/sound_ambient)**0.5*(0.6+(0.2/(0.2+Str_m) * \
        np.exp(-0.3*(theta_m+(Str_m/(1+Str_m))-2.7)**2)))

    #Calculation of the Source Strengh Function (FV)
    FV = ((Velocity_mixed-Velocity_aircraft)/sound_ambient)**velocity_exponent * \
        ((Velocity_mixed+Velocity_aircraft)/sound_ambient)**(1-velocity_exponent)

    #Determination of the noise model coefficients
    Z1 = -30*((1.8*theta_m/np.pi)-0.6)**2
    Z2 = -9 -4*((Velocity_primary-Velocity_secondary)/sound_ambient)-38*((1.8*theta_m/np.pi)-0.6)**3 + \
        30*(0.6-np.log10(1+Area_secondary/Area_primary))*(1.8*theta_m/np.pi - 0.6)
    Z3 = 1-0.4*((1.8*theta_m/np.pi)-0.6)**2
    Z4 = 0.44-0.5/np.exp(((4.5*theta_m/np.pi)-4)**2) + 0.2*Velocity_primary/sound_ambient - \
        0.7*Velocity_mixed/sound_ambient - 0.2*np.log10((1+Area_secondary)/Area_primary) + \
        0.05*(XBPR)*np.exp(-5*(theta_m-2.4)**2)
    Z5 = 34 + 81*theta_m/np.pi - 20*((1.8*theta_m/np.pi)-0.6)**3
    Z6 = 108 + 37.8*theta_m/np.pi + 5*Velocity_mixed*(Velocity_primary-Velocity_secondary)/(sound_ambient**2) - \
        np.exp(-5*(theta_m-1.8)**2) + 7*Velocity_mixed/sound_ambient*(1-0.4*(Velocity_primary/sound_ambient) * \
        np.exp(-0.7*np.abs(Str_m-0.8))) / np.exp(8*(theta_m-2.4)**2) + 0.8*(XBPR)*np.exp(theta_m-2.3-Velocity_mixed/sound_ambient) + \
        DSPL_m + EX_m

    #Determination of Sound Pressure Level for the mixed jet component
    SPL_m = (Z1*np.log10(FV)+Z2)*(np.log10(Str_m)-Z3*np.log10(FV)-Z4)**2 + Z5*np.log10(FV) + Z6

    return SPL_m 
