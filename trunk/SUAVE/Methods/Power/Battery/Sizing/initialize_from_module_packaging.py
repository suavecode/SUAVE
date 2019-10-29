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
    cell_mass                    = battery.cell.mass
    amp_hour_rating              = battery.cell.nominal_capacity 
    nominal_voltage              = battery.cell.nominal_voltage        
    mass                         = cell_mass* battery.module_config[0] * battery.module_config[1] 
    
    battery.mass_properties.mass = mass 
    battery.specific_energy      = (amp_hour_rating*Units.Wh*nominal_voltage)/mass * Units.Wh/Units.kg
    battery.max_energy           = mass*battery.specific_energy
    battery.max_voltage          = nominal_voltage * battery.module_config[0]
    
    
