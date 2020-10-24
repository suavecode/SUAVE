## @ingroupMethods-Noise-Fidelity_One-Engine
# external+plug_effect.py
# 
# Created:  Jul 2015, C. Ilario
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------        
#   Imports
# ---------------------------------------------------------------------- 
from SUAVE.Core  import Data
import numpy as np

# ----------------------------------------------------------------------        
#   External Plug Effect
# ---------------------------------------------------------------------- 

## @ingroupMethods-Noise-Fidelity_One-Engine
def external_plug_effect(Velocity_primary,Velocity_secondary, Velocity_mixed, Diameter_primary,
                         Diameter_secondary,Diameter_mixed, Plug_diameter, sound_ambient, theta_p,theta_s,theta_m):
    """This function calculates the adjustments, in decibels, to be added to the predicted jet noise levels due to
    external plugs in coaxial jets.
    
    Assumptions:
        N/A

    Source:
        N/A

    Inputs: 
        Velocity_primary      [m/s]
        Velocity_secondary    [m/s]
        Velocity_mixed        [m/s]
        Diameter_primary      [m]
        Diameter_secondary    [m]
        Diameter_mixed        [m]
        Plug_diameter         [m]
        sound_ambient         [dB]
        theta_p               [rad]
        theta_s               [rad]
        theta_m               [rad]
    
    Outputs: 
        PG_p        [dB]
        PG_s        [dB]
        PG_m        [dB]

    Properties Used:
        N/A  
    """

    # Primary jet
    PG_p = 0.1*(Velocity_primary/sound_ambient)*(10-(18*theta_p/np.pi))*Plug_diameter/Diameter_primary
    
    # Secondary jet
    PG_s = 0.1*(Velocity_secondary/sound_ambient)*(6-(18*theta_s/np.pi))*Plug_diameter/Diameter_secondary
    
    # Mixed jet
    PG_m = 0.1*(Velocity_primary*Velocity_mixed/(sound_ambient**2))*(9-(18*theta_m/np.pi))*Plug_diameter/Diameter_mixed
    
    # Pack Results 
    jet_plug_effects = Data()
    jet_plug_effects.PG_p = PG_p
    jet_plug_effects.PG_s = PG_s
    jet_plug_effects.PG_m = PG_m 

    return jet_plug_effects
