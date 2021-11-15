## @ingroup Methods-Cooling-Cryocooler-Cooling
# Cryocooler_model.py
# 
# Created:  Nov 2019, K.Hamilton
# Modified: Nov 2021,   S. Claridge

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Units, Data

# ----------------------------------------------------------------------
#   Cryocooler Model
# ----------------------------------------------------------------------

## @ingroup Methods-Cooling-Cryocooler-Cooling
# def cryocooler_model(max_power, cooler_type, cryo_temp, amb_temp=292.2):
def cryocooler_model(self, cooling_power, cryo_temp, amb_temp):
    """ Calculate the power required by the cryocooler based on the cryocooler type, the required cooling power, and the temperature conditions.
    
    Assumptions:
        Based on mass data for Cryomech cryocoolers as per the datasheets for ground based non-massreduced coolers available via the cryomech website: https://www.cryomech.com/cryocoolers/.
        The mass is calculated for the requested power level, the cryocooler should be sized for the maximum power level required as its mass will not change during the flight.
        The efficiency scales with required cooling power and temperature only.
        The temperature difference and efficiency are taken not to scale with ambient temperature. This should not matter in the narrow range of temperatures in which aircraft operate, i.e. for ambient temperatures between -50 and 50 C.
        
    Source: 
        https://www.cryomech.com/cryocoolers/
        
    Inputs:
        max_power -     cooling power required of the cryocooler                                [watts]
        cryo_temp -     cryogenic output temperature required                                   [kelvin]
        amb_temp -      ambient temperature the cooler will reject heat to, defaults to 19C     [kelvin]
        cooler_type -   cryocooler type used.   
    
    Outputs:
        output - a data dictionary with fields:
            input_power -   electrical input power required by the cryocooler         [watts]
            mass -          mass of the cryocooler and supporting components          [kilogram]
            coolerName -    Name of cooler type as a string
               
    Properties Used:
        N/A
    """ 

    # process
    # Initialise variables as null values
    # coolerName =    None    # Cryocooler type name
    # tempMinRT =     None    # Minimum temperature achievable by this type of cooler when rejecting to an ambient temperature of 19C (K)
    # tempMin =       None    # Updated minimum achievable temperature based on the supplied ambient temperature (K)
    # eff =           None    # Efficiency function. This is a line fit from a survey of Cryomech coolers in November 2019
    # input_power =   None    # Electrical input power (W)
    # mass =          None    # Total cooler mass function. Fit from November 2019 Cryomech data. (kg)

    # unpack cryocooler properties
    cooler_type     = self.cooler_type

    # Prevent unrealistic temperature changes.
    if np.amin(cryo_temp) < 1.:
        cryo_temp = np.maximum(cryo_temp, 5.)
        print("Warning: Less than zero kelvin not possible, setting cryogenic temperature target to 5K.")

    # Warn if ambient temperature is very low.
    if np.amin(amb_temp) < 200.:
        print("Warning: Suprisingly low ambient temperature, check altitude.")

    # Calculate the shift in achievable minimum temperature based on the the ambient temperature (temp_amb) and the datasheet operating temperature (19C, 292.15K)
    tempOffset = 292.15 - amb_temp

    # calculate the required temperature difference the cryocooler must produce.
    tempDiff = amb_temp-cryo_temp
    # Disable if the target temperature is greater than the ambient temp. Technically cooling like this is possible, however there are better cooling technologies to use if this is the required scenario.
    if np.amin(tempDiff) < 0.:
        tempDiff = np.maximum(tempDiff, 0.)
        print("Warning: Temperature conditions are not well suited to cryocooler use. Cryocooler disabled.")

    # Set the parameters of the cooler based on the cooler type and the operating conditions. The default ambient operating temperature (19C) is used as a base.
    if cooler_type ==   'fps':
        coolerName =    "Free Piston Stirling"
        tempMinRT =     35.0
        tempMin =       tempMinRT - tempOffset
        eff =           0.0014*(cryo_temp-tempMin)   
        input_power =   cooling_power/eff
        mass =          0.0098*input_power+1.0769

    elif cooler_type == 'GM':
        coolerName =    "Gifford McMahon"
        tempMinRT =     5.4
        tempMin =       tempMinRT - tempOffset
        eff =           0.0005*(cryo_temp-tempMin)
        input_power =   cooling_power/eff
        mass =          0.0129*input_power+63.08

    elif cooler_type == 'sPT':
        coolerName =    "Single Pulsetube"
        tempMinRT =     16.0
        tempMin =       tempMinRT - tempOffset
        eff =           0.0002*(cryo_temp-tempMin)
        input_power =   cooling_power/eff
        mass =          0.0079*input_power+51.124

    elif cooler_type == 'dPT':
        coolerName =    "Double Pulsetube"
        tempMinRT =     8.0
        tempMin =       tempMinRT - tempOffset
        eff =           0.00001*(cryo_temp-tempMin)
        input_power =   cooling_power/eff
        mass =          0.0111*input_power+73.809

    else:
        print("Warning: Unknown Cryocooler type")
        coolerName =    "Unknown"

    # Warn if the cryogenic temperature is unachievable
    diff = cryo_temp - tempMin
    if np.amin(diff) < 0.0:
        eff =           0.0
        input_power =   None
        mass =          None
        print("Warning: The required cryogenic temperature of " + str(cryo_temp) + " is not achievable using a " + cooler_type + " cryocooler at an ambient temperature of " + str(amb_temp) + ". The minimum temperature achievable is " + str(tempMin))
            
    # packup outputs
    self.name           = coolerName
  
    return [input_power, mass]
