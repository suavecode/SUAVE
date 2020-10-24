## @ingroupMethods-Noise-Fidelity_One-Engine
# primary_noise_component.py
# 
# Created:  Jul 2015, C. Ilario
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------   

import numpy as np

# ----------------------------------------------------------------------        
#   Primary Noise Component
# ----------------------------------------------------------------------  

## @ingroupMethods-Noise-Fidelity_One-Engine
def primary_noise_component(SPL_p,Velocity_primary,Temperature_primary,R_gas,theta_p,DVPS,sound_ambient,Velocity_secondary,Velocity_aircraft,Area_primary,Area_secondary,DSPL_p,EX_p,Str_p):
    """This function calculates the noise contribution of the primary jet component
    
        Assumptions:
        Empirical based procedure.
    
    Source: 
        None
        
    Inputs:
        noise_data     - SUAVE type vehicle

    Outputs:
        OASPL          - Overall Sound Pressure Level            [dB]
        PNL            - Perceived Noise Level                   [dB]
        PNL_dBA        - Perceived Noise Level A-weighted level  [dBA]
        EPNdB_takeoff  - Takeoff Effective Perceived Noise Level [EPNdB]
        EPNdB_landing  - Landing Effective Perceived Noise Level [EPNdB]  
    
    Properties Used:
        N/A  
       
    """      

    # Flow parameters of the primary jet
    sound_primary    = np.float(np.sqrt(1.4*R_gas*Temperature_primary))
    Mach_primary_jet = np.float(Velocity_primary/sound_primary)

    # Calculation of the velocity exponent
    for i in range (0,23):
        if theta_p[i] <= 2.2:
            velocity_exponent = 1.56
        elif theta_p[i] > 2.2:
            velocity_exponent = 1.5*np.exp(-10*(theta_p[i] - 2.2)**2)

        # Calculation of the Source Strengh Function (FV)
        FV = Mach_primary_jet*(np.float(DVPS)/sound_ambient)**0.6*(np.float((Velocity_primary+Velocity_secondary))/sound_ambient)**0.4* \
        (np.float(np.abs(Velocity_primary-Velocity_aircraft)/Velocity_primary))**velocity_exponent

        # Determination of the noise model coefficients
        Z1 = -18*((1.8*theta_p[i]/np.pi)-0.6)**2
        Z2 = -18-18*((1.8*theta_p[i]/np.pi)-0.6)**2
        Z3 = 0.0
        Z4 = -0.1 - 0.75*((Velocity_primary-Velocity_secondary-Velocity_aircraft)/sound_ambient) * \
            ((1.8*theta_p[i]/np.pi)-0.6)**3. + 0.8*(0.6-np.log10(1+Area_secondary/Area_primary))
        Z5 = 50 + 20*np.exp(-(theta_p[i]-2.6)**2.)
        Z6 = 94 + 46*np.exp(-(theta_p[i]-2.5)**2.) - 26.*(0.6-np.log10(1+Area_secondary/Area_primary))/ \
            np.exp(5*(theta_p[i]-2.3)**2) + DSPL_p[i] + EX_p

        # Determination of Sound Pressure Level for the primary jet component
        SPL_p[i] = (Z1*np.log10(FV)+Z2) * (np.log10(Str_p[i])-Z3*np.log10(FV)-Z4)**2 + Z5*np.log10(FV) + Z6

    return SPL_p

