## @ingroupMethods-Noise-Fidelity_One-Engine
# jet_installation_effect.py
# 
# Created:  Jul 2015, C. Ilario
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------        
#   Imports
# ---------------------------------------------------------------------- 

import numpy as np

# ----------------------------------------------------------------------        
#   Jet Installation Effect
# ---------------------------------------------------------------------- 

## @ingroupMethods-Noise-Fidelity_One-Engine
def jet_installation_effect (Xe,Ye,Ce,theta_s,Diameter_mixed):
    """This function calculates the installation effect, in decibels, to be added to the predicted secondary jet noise level."""
    
    #Ce = wing chord lenght at the engine location - as figure 7.3 of the SAE ARP 876D
    #Xe = fan exit location downstream of the leading edge (Xe<Ce) - as figure 7.3 of the SAE ARP 876D
    #Ye = separation distance from the wing chord line to nozzle lip - as figure 7.3 of the SAE ARP 876D

    #Instalation effect
    INST_s=0.5*((Ce-Xe)**2/(Ce*Diameter_mixed))*(np.exp(-Ye/Diameter_mixed)*((1.8*theta_s/np.pi))-0.6)**2

    #The magnitude of the installation effect is between 0 to 2.5 dB.
    for i in range (0,24):
        if INST_s[i]>2.5:
            INST_s[i]=2.5

    return (INST_s)