# UAV_weights.py
# 
# Created:  Jan 2016, E. Botero
# Modified: 

#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------


import SUAVE
from SUAVE.Core import Units, Data

def empty(vehicle):
    
    """
    Structural Weight correlation from all 415 samples of fixed-wing UAVs and sailplanes
    Equation 3.16 from 'Design of Solar Powered Airplanes for Continuous Flight' by Andre Noth
    Relatively valid for a wide variety of vehicles, may be optimistic
    Assumes a 'main wing' is attached
    
    """
    
    # Unpack
    S     = vehicle.reference_area
    AR    = vehicle.wings['main_wing'].aspect_ratio
    Earth = SUAVE.Attributes.Planets.Earth()
    g     = Earth.sea_level_gravity
    
    
    # Airframe weight
    Waf = 0.654* (5.58*(S**1.59)*(AR**0.71))/g # All Samples # Matthew: Fudge Factor of 0.654 added for Aquila
    
    # Pack
    weight = Data()
    weight.empty = Waf
    
    vehicle.wings['main_wing'].mass_properties.mass = Waf/g
    
    return weight