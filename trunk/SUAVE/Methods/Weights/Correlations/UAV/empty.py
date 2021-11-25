## @ingroup Methods-Weights-Correlations-UAV
# empty.py
# 
# Created:  Jan 2016, E. Botero
# Modified: 

#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
import numpy as np

from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Empty
# ----------------------------------------------------------------------

## @ingroup Methods-Weights-Correlations-UAV
def empty(vehicle,settings=None):
    """ This computes the weigt of a UAV   
    
    Assumptions:
        Assumes a 'main wing' is attached

    Source:
        Structural Weight correlation from all 415 samples of fixed-wing UAVs and sailplanes
        Equation 3.16 from 'Design of Solar Powered Airplanes for Continuous Flight' by Andre Noth
        Relatively valid for a wide variety of vehicles, may be optimistic 

    Inputs:
        S                [meters**2]
        AR               [dimensionless]
        
    Outputs:
        weight           [kilograms]

    Properties Used:
        N/A
    """    
    
    # Unpack
    S     = vehicle.reference_area
    AR    = vehicle.wings['main_wing'].aspect_ratio
    Earth = SUAVE.Attributes.Planets.Earth()
    g     = Earth.sea_level_gravity
    
    
    # Airframe weight
    Waf = (5.58*(S**1.59)*(AR**0.71))/g # All Samples
    #Waf = (0.44*(S**1.55)*(AR**1.30))/g  # Top 19
    
    # Pack
    weight = Data()
    weight.empty = Waf
    
    return weight