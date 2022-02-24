## @ingroup Components-Energy-Cooling
# Cryogenic_Heat_Exchanger.py
#
# Created:  Feb 2020, K. Hamilton

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE

import numpy as np
from scipy.optimize import fsolve

from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Methods.Cooling.Cryogen.Consumption import Coolant_use

# ----------------------------------------------------------------------
#  Cryogenic Heat Exchanger Component
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Converters
class Cryogenic_Heat_Exchanger(Energy_Component):
    """This provides output values for a heat exchanger used to cool cryogenic components
    
    Assumptions:
    None
    
    Source:
    N/A
    """
    
    def __defaults__(self):
        """This sets the default values for the component to function.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """         
        
        self.tag = 'Cryogenic_Heat_Exchanger'
        
        #-----setting the default values for the different components
        self.cryogen                        = SUAVE.Attributes.Cryogens.Liquid_H2()
        self.cryogen_inlet_temperature      =    300.0      # [K]
        self.cryogen_outlet_temperature     =    300.0      # [K]
        self.cryogen_pressure               = 100000.0      # [Pa]
        self.cryogen_is_fuel                =      0.0      # Proportion of cryogen that is burned as fuel. Assumes the cryogen is the same as the fuel, e.g. that both are hydrogen.
    
    def energy_calc(self,cooling_power, conditions):
        """ This calculates the mass of cryogen required to achieve the desired cooling power given the temperature of the cryogen supplied, and the desired temperature of the cryogenic equipment.

        Assumptions:
        Perfect thermal conduction of the cryogen to the cooled equipment.

        Source:
        N/A

        Inputs:
        cryogenic_heat_exchanger.
            cryogen_inlet_temperature       [K]
            cryogen_outlet_temperature      [K]
            cryogen_pressure                [Pa]

        Outputs:
        cryogen_mass_flow                   [kg/s]

        Properties Used:
        
        """         
        # unpack the values from self
        temp_in         = self.cryogen_inlet_temperature 
        temp_out        = self.cryogen_outlet_temperature
        pressure        = self.cryogen_pressure
        cryogen         = self.cryogen        
        amb_pressure    = conditions.freestream.pressure

        # If the heat exchanger does not vent to ambient, set the system pressure.
        vent_pressure   = pressure

        # calculate the cryogen mass flow
        mdot = Coolant_use(cryogen,temp_in,temp_out,cooling_power,vent_pressure)
    
        return mdot
        
    