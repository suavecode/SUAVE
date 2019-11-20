## @ingroup Components-Energy-Storages-Batteries
# Battery.py
# 
# Created:  Nov 2014, M. Vegh
# Modified: Feb 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports

from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Methods.Power.Battery.Discharge.datta_discharge import datta_discharge
from SUAVE.Methods.Power.Battery.Discharge.thevenin_discharge  import thevenin_discharge
from SUAVE.Methods.Power.Battery.Charge.datta_charge import datta_charge
from SUAVE.Methods.Power.Battery.Charge.thevenin_charge  import thevenin_charge


# ---------------------------------------------------------------- ------
#  Battery
# ----------------------------------------------------------------------    

## @ingroup Components-Energy-Storages-Batteries
class Battery(Energy_Component):
    """
    Energy Component object that stores energy. Contains values
    used to indicate its discharge characterics, including a model
    that calculates discharge losses
    """
    def __defaults__(self):
        self.mass_properties.mass     = 0.0
        self.energy_density           = 0.0
        self.current_energy           = 0.0
        self.initial_temperature      = 20.0
        self.current_capacitor_charge = 0.0
        self.resistance               = 0.07446 # base internal resistance of battery in ohms
        self.max_energy               = 0.0
        self.max_power                = 0.0
        self.max_voltage              = 0.0
        self.datta_discharge_model    = datta_discharge
        self.thevenin_discharge_model = thevenin_discharge
        self.datta_charge_model       = datta_charge
        self.thevenin_charge_model    = thevenin_charge        
        self.ragone                   = Data()
        self.ragone.const_1           = 0.0     # used for ragone functions; 
        self.ragone.const_2           = 0.0     # specific_power=ragone_const_1*10^(specific_energy*ragone_const_2)
        self.ragone.lower_bound       = 0.0     # lower bound specific energy for which ragone curves no longer make sense
        self.ragone.i   = 0.0
        
    def energy_discharge(self,numerics,fidelity = 1):
        if fidelity == 1:
            self.datta_discharge_model(self, numerics)
        elif fidelity == 2:
            self.thevenin_discharge_model(self, numerics)
        else:
            assert AttributeError("Fidelity must be '1' (datta discharge model) or '2' (thevenin discharge model")
        return  
    
    def energy_charge(self,numerics,fidelity = 1):
        if fidelity == 1:
            self.datta_charge_model(self, numerics)
        elif fidelity == 2:
            self.thevenin_charge_model(self, numerics)
        else:
            assert AttributeError("Fidelity must be '1' (datta discharge model) or '2' (thevenin discharge model")
        return  