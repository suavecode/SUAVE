# noise_landing_gear.py
# 
# Created:  Jun 2015, Carlos
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    
import numpy as np
from SUAVE.Core import Units

# ----------------------------------------------------------------------
# Compute the main and nose landing gear noise
# ----------------------------------------------------------------------

def noise_landing_gear (D,H,wheels,M,velocity,phi,theta,distance,frequency):
    """ SUAVE.Methods.Noise.Fidelity_One.noise_landing_gear(D,H,wheels,M,velocity,phi,theta,distance,frequency):
            Calculates the Landing gear 1/3 octave band sound pressure level and overall sound pressure level
            for a tyre diameter D, a strut length H and WHEELS number of  wheels per unit.

            Inputs:
                    D                          - Landing gear tyre diameter [ft]
                    H                          - Lading gear strut length [ft]
                    wheels                     - Number of wheels per unit
                    M                          - Mach number
                    velocity                   - Aircraft speed [kts]
                    phi                        - Azimuthal angle [rad]
                    theta                      - Polar angle [rad]
                    distance                   - Distance from airplane to observer, evaluated at retarded time [ft]
                    frequemcy                  - Frequency array [Hz]



            Outputs: One Third Octave Band SPL [dB]
                SPL                              - Sound Pressure Level of the landing gear [dB]
                OASPL                            - Overall Sound Pressure Level of the landing gear [dB]

            Assumptions:
                Correlation based."""


    #Process
    
    velocity_fts = velocity/Units.ft
    velocity_kts = velocity/Units.knots

    if (wheels==1 or wheels==2):
        G1 = 13+np.log10(4.5*((frequency*D/(velocity_fts*(1-M*np.cos(theta))))**2)* \
            (12.5+((frequency*D/(velocity_fts*(1-M*np.cos(theta))))**2))**-2.25)
        G2 = (13+np.log10(2.0*(frequency*D/(velocity_fts*(1-M*np.math.cos(theta)))**2.0))* \
            (30+(frequency*D/(velocity_fts*(1-M*np.cos(theta))))**8)**-1*(0.34*H/D))* \
            (np.math.sin(phi))**2
    elif wheels==4:
        G1 = 12+np.log10(frequency*D/(velocity_fts*(1-M*np.cos(theta))))**2 \
        *(0.4+(frequency*D/(velocity_fts*(1-M*np.cos(theta))))**2)**(-1.6)
        G2 = (12+np.log10(7.0*(frequency*D/(velocity_fts*(1-M*np.cos(theta))))**3.0 * \
            (1.06+(frequency*D/(velocity_fts*(1-M*np.cos(theta))))**2)**(-3.0)*(1)))*(np.sin(phi))**2


    G3    = 12.79+np.log10(0.34*H/D)*(np.sin(phi))**2
    SPL   = 60.*np.log10(velocity_kts/194.0)+20.*np.log10(D/distance)+10.*np.log10(10.0**G1+10.0**G2)
    OASPL = 60.*np.log10(velocity_kts/194.0)+20.*np.log10(D/distance)+10.*np.log10(10.0**12.52+10.0**G3)

    return(SPL)
