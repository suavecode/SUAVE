## @ingroupMethods-Noise-Fidelity_One-Airframe
# noise_trailing_edge_flap.py
# 
# Created:  Jul 2015, C. Ilario
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------        
#   Imports
# ---------------------------------------------------------------------- 

import numpy as np
from SUAVE.Core import Units

# ----------------------------------------------------------------------
# Compute the trailing edge flap noise
# ----------------------------------------------------------------------

## @ingroupMethods-Noise-Fidelity_One-Airframe
def noise_trailing_edge_flap(Sf,cf,deltaf,slots,velocity,M,phi,theta,distance,frequency):
    """ This calculates the noise from the flap trailing edge as a 1/3 octave band sound pressure level.

    Assumptions:
        Correlation based.
        
    Inputs:
        Sf         - Flap area                                                      [sq.ft]
        cf         - Flap chord                                                     [ft]
        deltaf     - Flap deflection                                                [rad]
        slots      - Number of slots                                                (Flap type)
        velocity   - Aircraft speed                                                 [kts]
        M          - Mach number                                                    [Unitless]
        phi        - Azimuthal angle                                                [rad]
        theta      - Polar angle                                                    [rad]
        distance   - Distance from airplane to observer, evaluated at retarded time [ft]
        frequency  - Frequency array                                                [Hz]
        
    Outputs: One Third Octave Band SPL                                              [dB]
        SPL                
 
    Properties Used:
        N/A 
    
    """

    # Process
    G      = np.zeros(24)

    test   = frequency*cf/(velocity/Units.ft*(1-M*np.cos(theta)))

    if (slots==1 or slots==2):
        for i in range (0,24):
            if (test[i]<2):
                G[i] = 99+10*np.log10(test[i])
            elif (test[i]<20):
                G[i] = 103.82-6*np.log10(test[i])
            else:
                G[i] = 135.04-30*np.log10(test[i])

    elif slots==3:
        for i in range(0,24):
            if(test[i]<2):
                G[i] = 99+10*np.log10(test[i])
            elif (test[i]<75):
                G[i] = 102.61-2*np.log10(test[i])
            else:
                G[i] = 158.11-30*np.log10(test[i])

    G = np.transpose(G)
    
    if theta+deltaf>=np.pi:
        directivity = 0.0
    else:     
        directivity = 20.0*np.log10(np.sin(theta)* (np.cos(phi))**2 * np.sin(theta+deltaf))

    SPL = G+10*np.log10(Sf*(np.sin(deltaf))**2/(distance**2))+ \
        60*np.log10((velocity/Units.kts)/100.0)+directivity

    return SPL 
