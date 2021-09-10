## @ingroup Methods-Power-Battery-Sizing
# initialize_from_mass.py
# 
# Created:  ### ####, M. Vegh
# Modified: Feb 2016, E. Botero
#           Aug 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------
## @ingroup Methods-Power-Battery-Sizing
def initialize_from_mass(battery, SOC_start = 1, SOC_cutoff = 0.15 ):
    """
    Calculate the max energy and power based of the mass
    Assumptions:
    A constant value of specific energy and power

    Inputs:
    mass              [kilograms]
    battery.
      specific_energy [J/kg]               
      specific_power  [W/kg]

    Outputs:
     battery.
       max_energy
       max_power
       mass_properties.
        mass


    """    
    mass                         = battery.mass_properties.mass 
    n_cells                      = int(mass/battery.cell.mass)
    n_series                     = int(battery.max_voltage/battery.cell.max_voltage)
    n_parallel                   = int(n_cells/n_series)
    battery.max_energy           = mass*battery.specific_energy* SOC_start 
    battery.min_energy           = mass*battery.specific_energy* SOC_cutoff
    battery.max_power            = mass*battery.specific_power
    battery.initial_max_energy   = battery.max_energy    
    battery.pack_config.series   = n_series
    battery.pack_config.parallel = n_parallel      
    battery.pack_config.total    = n_cells