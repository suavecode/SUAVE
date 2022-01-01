## @ingroup Components-Energy-Cooling
# Cryocooler.py
# 
# Created:  Feb 2020,   K.Hamilton
# Modified: Nov 2021,   S. Claridge

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component
import numpy as np

# ----------------------------------------------------------------------
#  Cryocooler
# ----------------------------------------------------------------------    
## @ingroup Components-Energy-Cooling-Cryocooler
class Cryocooler(Energy_Component):
    
    """
    Cryocooler provides cooling power to cryogenic components.
    Energy is used by this component to provide the cooling, despite the cooling power provided also being an energy inflow.
    """
    def __defaults__(self):
        
        # Initialise cryocooler properties as null values
        self.cooler_type        =  ''
        self.rated_power        =  0.0
        self.min_cryo_temp      =  0.0 
        self.ambient_temp       =  300.0

    def energy_calc(self, cooling_power, cryo_temp, amb_temp):

        """ Calculate the power required by the cryocooler based on the cryocooler type, the required cooling power, and the temperature conditions.
    
    Assumptions:
        Based on mass data for Cryomech cryocoolers as per the datasheets for ground based non-massreduced coolers available via the cryomech website: https://www.cryomech.com/cryocoolers/.
        The mass is calculated for the requested power level, the cryocooler should be sized for the maximum power level required as its mass will not change during the flight.
        The efficiency scales with required cooling power and temperature only.
        The temperature difference and efficiency are taken not to scale with ambient temperature. This should not matter in the narrow range of temperatures in which aircraft operate, i.e. for ambient temperatures between -50 and 50 C.
        
    Source: 
        https://www.cryomech.com/cryocoolers/
        
    Inputs:

            cooling_power -     cooling power required of the cryocooler                                [watts]
            cryo_temp -         cryogenic output temperature required                                   [kelvin]
            amb_temp -          ambient temperature the cooler will reject heat to, defaults to 19C     [kelvin]
            cooler_type -       cryocooler type used
 
    
    Outputs:

            input_power -   electrical input power required by the cryocooler         [watts]
            mass -          mass of the cryocooler and supporting components          [kilogram]

    Properties Used:
        N/A
    """      
        # Prevent unrealistic temperature changes.
        if np.amin(cryo_temp) < 1.:

            cryo_temp = np.maximum(cryo_temp, 5.)
            print("Warning: Less than zero kelvin not possible, setting cryogenic temperature target to 5K.")

        # Warn if ambient temperature is very low.
        if np.amin(amb_temp) < 200.:

            print("Warning: Suprisingly low ambient temperature, check altitude.")

        # Calculate the shift in achievable minimum temperature based on the the ambient temperature (temp_amb) and the datasheet operating temperature (19C, 292.15K)
        temp_offset = 292.15 - amb_temp

        # Calculate the required temperature difference the cryocooler must produce.
        temp_diff = amb_temp-cryo_temp

        # Disable if the target temperature is greater than the ambient temp. Technically cooling like this is possible, however there are better cooling technologies to use if this is the required scenario.
        if np.amin(temp_diff) < 0.:

            temp_diff = np.maximum(temp_diff, 0.)
            print("Warning: Temperature conditions are not well suited to cryocooler use. Cryocooler disabled.")

        # Set the parameters of the cooler based on the cooler type and the operating conditions. The default ambient operating temperature (19C) is used as a base.
        if self.cooler_type ==   'fps': #Free Piston Stirling

            temp_minRT      =  35.0      # Minimum temperature achievable by this type of cooler when rejecting to an ambient temperature of 19C (K)
            temp_min        =  temp_minRT - temp_offset   # Updated minimum achievable temperature based on the supplied ambient temperature (K)
            eff             =  0.0014*(cryo_temp-temp_min) # Efficiency function. This is a line fit from a survey of Cryomech coolers in November 2019  
            input_power     =  cooling_power/eff           # Electrical input power (W)
            mass            =  0.0098*input_power+1.0769   # Total cooler mass function. Fit from November 2019 Cryomech data. (kg)

        elif self.cooler_type == 'GM': #Gifford McMahon

            temp_minRT      =  5.4
            temp_min        =  temp_minRT - temp_offset
            eff             =  0.0005*(cryo_temp-temp_min)
            input_power     =  cooling_power/eff
            mass            =  0.0129*input_power+63.08 

        elif self.cooler_type == 'sPT': #Single Pulsetube

            temp_minRT      =  16.0
            temp_min        =  temp_minRT - temp_offset
            eff             =  0.0002*(cryo_temp-temp_min)
            input_power     =  cooling_power/eff
            mass            =  0.0079*input_power+51.124  

        elif self.cooler_type == 'dPT': #Double Pulsetube

            temp_minRT      =  8.0
            temp_min        =  temp_minRT - temp_offset
            eff             =  0.00001*(cryo_temp-temp_min)
            input_power     =  cooling_power/eff
            mass            =  0.0111*input_power+73.809  

        else:

            print("Warning: Unknown Cryocooler type")
            return[0.0,0.0]

        # Warn if the cryogenic temperature is unachievable
        diff = cryo_temp - temp_min

        if np.amin(diff) < 0.0:

            eff         =   0.0
            input_power =   None
            mass        =   None

            print("Warning: The required cryogenic temperature of " + str(cryo_temp) + " is not achievable using a " + self.cooler_type + " cryocooler at an ambient temperature of " + str(amb_temp) + ". The minimum temperature achievable is " + str(temp_min))

        self.mass_properties.mass     = mass
        self.rated_power              = input_power

        return [input_power, mass]



