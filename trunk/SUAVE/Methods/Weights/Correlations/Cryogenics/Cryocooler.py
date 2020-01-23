## @ingroup Methods-Weights-Correlations-Cryogenics 
# Cryocooler.py
# 
# Created:  Nov 2019, K.Hamilton

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units, Data

# ----------------------------------------------------------------------
#   Cryocooler
# ----------------------------------------------------------------------

## @ingroup Methods-Weights-Correlations-Cryogenics 
def cryocooler(max_power, cooler_type, cryo_temp, amb_temp=292.2):
    """ Calculate the weight of the cryocooler
    
    Assumptions:
        Based on mass data for Cryomech cryocoolers as per the datasheets for ground based non-massreduced coolers available via the cryomech website: https://www.cryomech.com/cryocoolers/
        
    Source: 
        https://www.cryomech.com/cryocoolers/
        
    Inputs:
        max_power - maximum cooling power required of the cryocooler                            [watts]
        cryo_temp - cryogenic output temperature required                                       [kelvin]
        amb_temp - ambient temperature the cooler will reject heat to, defaults to 19C          [kelvin]
        cooler_type - cryocooler type used.   
    
    Outputs:
        output - a data dictionary with fields:
            mass - mass of the cryocooler and supporting components                 [kilogram]
            input_power - electrical input power required by the cryocooler         [watts]
            coolerName - Name of cooler type as a string
               
    Properties Used:
        N/A
    """ 

    # process
    # Initialise variables as null values
    coolerName =    None    # Cryocooler type name
    tempMin =       None    # Minimum temperature achievable by this type of cooler when rejecting to an ambient temperature of 19C (K)
    eff =           None    # Efficiency function. This is a line fit from a survey of all the coolers available from Cryomech in November 2019
    input_power =   None    # Electrical input power (W)
    mass =          None    # Total cooler mass function. Fit from November 2019 Cryomech data. (kg)

    # Set the parameters of the cooler based on the cooler type and the operating conditions. Presently only the default ambient operating temperature (19C) is supported.
    if cooler_type = 'fps':
        coolerName = "Free Piston Stirling"
        tempMin = 35.0
        eff = 0.0014*(cryo_temp-35.0)   
        input_power = max_power/eff
        mass = 0.0098*input_power+1.0769

    elif cooler_type = 'GM':
        coolerName = "Gifford McMahon"
        tempMin = 5.4
        eff = 0.0005*(cryo_temp-5.4)
        input_power = max_power/eff
        mass = 0.0129*input_power+63.08

    elif cooler_type = 'sPT':
        coolerName = "Single Pulsetube"
        tempMin = 16.0
        eff = 0.0002*(cryo_temp-16.0)
        input_power = max_power/eff
        mass = 0.0282*input_power+5.9442

    elif cooler_type = 'dPT':
        coolerName = "Double Pulsetube"
        tempMin = 8.0
        eff = 0.00001*(cryo_temp-8.0)
        input_power = max_power/eff
        mass = 0.0291*input_power+3.9345

    else:
        print("Warning: Unknown Cryocooler type")
        coolerName = "Unknown"

    if cryo_temp < tempMin:
        eff =           0.0
        input_power =   None
        mass =          None
        print("Warning: The required cryogenic temperature of " + str(cryo_temp) + " is not achievable using a " + coolerType + " cryocooler. The minimum temperature achievable is " + str(tempMin))
            
    # packup outputs
    output = Data()
    output.name =           coolerName
    output.mass =           mass
    output.input_power =    input_power
  
    return output