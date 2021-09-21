
## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
# Lithium_Ion.py
# 
# Created:  Nov 2014, M. Vegh
# Modified: Feb 2016, T. MacDonald
#           Sep 2021, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
from SUAVE.Core import Units, Data
from SUAVE.Components.Energy.Storages.Batteries  import Battery

# ----------------------------------------------------------------------
#  Lithium_Ion
# ----------------------------------------------------------------------    
## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
class Lithium_Ion(Battery):
    """
    Specifies discharge/specific energy characteristics specific tobytes
    lithium-ion batteries
    """
    def __defaults__(self):

        self.tag               = 'Lithium_Ion_Battery_Cell'         
        self.cell              = Data()   
        self.module            = Data()        
        self.pack_config       = Data()
        self.module_config     = Data()
        self.cooling_fluid     = Data()
        
        self.cell.charging_SOC_cutoff                      = 1. 
        self.cell.charging_current                         = 3.0        # [Amps]
                         
        self.heat_transfer_coefficient                     = 35.     # [W/m^2K] 
        self.heat_transfer_efficiency                      = 1.0       
        
        self.pack_config.series                            = 1
        self.pack_config.parallel                          = 1  
        self.pack_config.total                             = 1   
        self.module_config.total                           = 1       
        self.module_config.normal_count                    = 1       # number of cells normal to flow
        self.module_config.parallel_count                  = 1       # number of cells parallel to flow      
        self.module_config.normal_spacing                  = 0.02
        self.module_config.parallel_spacing                = 0.02
        

        self.cooling_fluid.tag                             = 'air'
        self.cooling_fluid.thermal_conductivity            = 0.0253 # W/mK
        self.cooling_fluid.specific_heat_capacity          = 1006   # K/kgK  
        self.cooling_fluid.discharge_air_cooling_flowspeed = 0.01   
        self.cooling_fluid.charge_air_cooling_flowspeed    = 0.01          
        
        # Default is LFP
        self.specific_energy    = 200.    *Units.Wh/Units.kg  # update this to match LFP
        self.specific_power     = 1.      *Units.kW/Units.kg  # update this to match LFP
        
        self.ragone.const_1     = 88.818  *Units.kW/Units.kg
        self.ragone.const_2     = -0.01533 /(Units.Wh/Units.kg)
        self.ragone.lower_bound = 60.     *Units.Wh/Units.kg
        self.ragone.upper_bound = 225.    *Units.Wh/Units.kg