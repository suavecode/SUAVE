## @ingroup Components-Energy-Converters
# Turboelectric.py
#
# Created:  Nov 2019, K. Hamilton
# Modified: Nov 2021,   S. Claridge
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
from SUAVE.Core import Units
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Attributes.Gases import Air
from SUAVE.Attributes.Propellants import Liquid_Natural_Gas
from SUAVE.Methods.Power.Turboelectric.Discharge import zero_fidelity

# ----------------------------------------------------------------------
#  Turboelectric Class
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Converters
class Turboelectric(Energy_Component):
    """This is a turboelectic component.
    
    Assumptions:
    None

    Source:
    None
    """    
    def __defaults__(self):
        """This sets the default values for the component to function.

        Assumptions:
        None

        Source:
        https://new.siemens.com/global/en/products/energy/power-generation/gas-turbines/sgt-a30-a35-rb.html

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """           
        self.propellant             = Liquid_Natural_Gas()
        self.oxidizer               = Air()
        self.number_of_engines      = 2.0                   # number of turboelectric machines, not propulsors
        self.efficiency             = .37                   # Approximate average gross efficiency across the product range.
        self.volume                 = 2.36    *Units.m**3.  # 3m long from RB211 datasheet. 1m estimated radius.
        self.rated_power            = 37400.0 *Units.kW
        self.mass_properties.mass   = 2500.0  *Units.kg     # 2.5 tonnes from Rolls Royce RB211 datasheet 2013.
        self.specific_power         = self.rated_power/self.mass_properties.mass
        self.mass_density           = self.mass_properties.mass/self.volume
        self.discharge_model        = zero_fidelity         # Simply takes the fuel specific power and applies an efficiency.
        
    def energy_calc(self,conditions,numerics):
        """This calls the assigned discharge method.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        see properties used

        Outputs:
        mdot     [kg/s] (units may change depending on selected model)

        Properties Used:
        self.discharge_model(self, conditions, numerics)
        """           
        
        mdot = self.discharge_model(self, conditions, numerics)
        return mdot  

    
