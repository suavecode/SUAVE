## @ingroupMethods-Noise-Fidelity_One-Airframe
# noise_clean_wing.py
# 
# Created:  Jun 2015, C. Ilario
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import numpy as np
from SUAVE.Core import Units

# ----------------------------------------------------------------------
# Compute the clean wing noise
# ----------------------------------------------------------------------

## @ingroupMethods-Noise-Fidelity_One-Airframe
def noise_clean_wing(S,b,ND,IsHorz,velocity,viscosity,M,phi,theta,distance,frequency):
    """ This computes the 1/3 octave band sound pressure level and the overall sound pressure level from the clean wing,
    for a wing with area S (sq.ft) and span b (ft).  ND is a constant set to 0 for clean wings and set to 1 for propeller
    airplanes, jet transports with numerous large trailing edge flap tracks, flaps extended, or slats extended. ISHORZ must be set to 1.
    This function can be used for the horizontal tail by inserting the appropriate tail area and span. For a vertical tail, its appropriate
    area and height are used and ISHORZ must be set to 0.
    
    Assumptions:
        Correlation based.  
    
    Source:
       SAE Model
       
    Inputs:
            S                          - Wing Area [sq.ft]
            b                          - Wing Span [ft]
            ND                         - Costant from the method
            IsHoriz                    - Costant from the method
            deltaw                     - Wing Turbulent Boundary Layer thickness [ft]
            velocity                   - Aircraft speed [kts]
            viscosity                  - Dynamic viscosity
            M                          - Mach number
            phi                        - Azimuthal angle [rad]
            theta                      - Polar angle [rad]
            distance                   - Distance from airplane to observer, evaluated at retarded time [ft]
            frequency                  - Frequency array [Hz]



    Outputs: One Third Octave Band SPL [dB]
        SPL                              - Sound Pressure Level of the clean wing [dB]
        OASPL                            - Overall Sound Pressure Level of the clean wing [dB]

    Properties Used:
        None
    
    """
 
    delta  = 0.37*(S/b)*(velocity/Units.ft*S/(b*viscosity))**(-0.2)

    if IsHorz==1:
        DIR = np.cos(phi)
    elif IsHorz==0:
        DIR = np.sin(phi)


    if DIR==0:
        SPL = np.zeros(24)
    else:

        fmax  = 0.1*(velocity/Units.ft)/(delta*(1-M*np.cos(theta))) 

        OASPL = 50*np.log10((velocity/Units.kts)/100.0)+10*np.log10(delta*b/(distance**2.0))+8*ND+ \
            20*np.log10(DIR*np.sin(theta)*np.cos(theta/2.0))+104.3

        SPL   = OASPL+10.0*np.log10(0.613*(frequency/fmax)**4*((frequency/fmax)**1.5+0.5)**(-4))-0.03*np.abs(((frequency/fmax)-1))**1.5

    return SPL
