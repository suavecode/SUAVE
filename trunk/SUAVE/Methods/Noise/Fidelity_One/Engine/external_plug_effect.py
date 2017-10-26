## @ingroupMethods-Noise-Fidelity_One-Engine
# external+plug_effect.py
# 
# Created:  Jul 2015, C. Ilario
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------        
#   Imports
# ---------------------------------------------------------------------- 

import numpy as np

# ----------------------------------------------------------------------        
#   External Plug Effect
# ---------------------------------------------------------------------- 

## @ingroupMethods-Noise-Fidelity_One-Engine
def external_plug_effect (Velocity_primary,Velocity_secondary, Velocity_mixed, Diameter_primary,Diameter_secondary,Diameter_mixed, Plug_diameter, sound_ambient, theta_p,theta_s,theta_m):
    """This function calculates the adjustments, in decibels, to be added to the predicted jet noise levels due to
    external plugs in coaxial jets."""

    #Primary jet
    PG_p = 0.1*(Velocity_primary/sound_ambient)*(10-(18*theta_p/np.pi))*Plug_diameter/Diameter_primary
    #Secondary jet
    PG_s = 0.1*(Velocity_secondary/sound_ambient)*(6-(18*theta_s/np.pi))*Plug_diameter/Diameter_secondary
    #Mixed jet
    PG_m = 0.1*(Velocity_primary*Velocity_mixed/(sound_ambient**2))*(9-(18*theta_m/np.pi))*Plug_diameter/Diameter_mixed

    return(PG_p,PG_s,PG_m)
