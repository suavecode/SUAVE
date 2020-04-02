## @ingroup Methods-Power-Battery-Sizing
# initialize_from_mass.py
# 
# Created:  ### ####, M. Vegh
# Modified: Feb 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Units
import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------
## @ingroup Methods-Power-Battery-Sizing
def initialize_from_module_packaging(battery):  
    amp_hour_rating              = battery.cell.nominal_capacity 
    nominal_voltage              = battery.cell.nominal_voltage        
    total_battery_assemply_mass  = battery.cell.mass * battery.module_config[0] * battery.module_config[1] 
    
    battery.mass_properties.mass = total_battery_assemply_mass
    battery.specific_energy      = (amp_hour_rating*nominal_voltage)/battery.cell.mass  * Units.Wh/Units.kg
    battery.max_energy           = total_battery_assemply_mass*battery.specific_energy
    battery.max_voltage          = battery.cell.max_voltage  * battery.module_config[0] 
    battery.initial_max_energy   = battery.max_energy    
    
    battery.charging_voltage     = battery.cell.charging_voltage * battery.module_config[0] 
    battery.charging_current     = battery.cell.charging_current * battery.module_config[1] 
