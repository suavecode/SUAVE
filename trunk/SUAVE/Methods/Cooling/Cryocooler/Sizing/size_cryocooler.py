## @ingroup Methods-Cooling-Cryocooler-Sizing
# Cryocooler.py
# 
# Created:  Feb 2020, K.Hamilton
# Modified: Nov 2021, S. Claridge

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units, Data
from SUAVE.Methods.Cooling.Cryocooler.Cooling.cryocooler_model import cryocooler_model

# ----------------------------------------------------------------------
#   Cryocooler
# ----------------------------------------------------------------------

## @ingroup Methods-Cooling-Cryocooler-Sizing
def size_cryocooler(cryocooler, max_power, cryo_temp, amb_temp=292.2):
    """ Calculate the cryocooler mass.
    
    Assumptions:
        See cryocooler_model
        
    Source: 
        https://www.cryomech.com/cryocoolers/
        
    Inputs:
        max_power -     cooling power required of the cryocooler                                [watts]
        cryo_temp -     cryogenic output temperature required at sizing                         [kelvin]
        amb_temp -      ambient temperature the cooler will reject heat to, defaults to 19C     [kelvin]
        cooler_type -   cryocooler type used.   
    
    Outputs:
        self.
            mass -      mass of the cryocooler      [kg]
               
    Properties Used:
        N/A
    """ 

    # Call the cryocooler model and run with the sizing parameters
    output = cryocooler_model(cryocooler, max_power, cryo_temp, amb_temp)

    # Pack up outputs
    cryocooler.mass_properties.mass     = output[1]
    cryocooler.rated_power              = output[0]