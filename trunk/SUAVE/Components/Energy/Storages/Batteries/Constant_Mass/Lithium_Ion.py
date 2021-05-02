## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
# Lithium_Ion.py
# 
# Created:  Nov 2014, M. Vegh
# Modified: Feb 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE
import numpy as np 
# package imports
from SUAVE.Core import Units , Data 
from SUAVE.Components.Energy.Storages.Batteries  import Battery

# ----------------------------------------------------------------------
#  Lithium_Ion
# ----------------------------------------------------------------------    
## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
class Lithium_Ion(Battery):
    """
    Specifies discharge/specific energy characteristics of the default 
    lithium-ion battery in SUAVE. 
    """
    def __defaults__(self):
        self.tag                                          = 'Lithium_Ion_Battery'
        self.chemistry                                    = 'Li_Generic'
        self.cell                                         = Data()  
        self.module                                       = Data()        
        self.pack_config                                  = Data()
        self.module_config                                = Data()
        self.cooling_fluid                                = Data()
                         
        self.specific_energy                              = 200.    * Units.Wh/Units.kg
        self.specific_power                               = 1.      * Units.kW/Units.kg
        self.ragone.const_1                               = 88.818  * Units.kW/Units.kg
        self.ragone.const_2                               = -.01533 / (Units.Wh/Units.kg)
        self.ragone.lower_bound                           = 60.     * Units.Wh/Units.kg
        self.ragone.upper_bound                           = 225.    * Units.Wh/Units.kg 
        
        self.specific_heat_capacity                       = 2000    # [J/kgK] 
        self.cell.specific_heat_capacity                  = 2000    # [J/kgK] 
        self.heat_transfer_coefficient                    = 35.     # [W/m^2K] 
        self.heat_transfer_efficiency                     = 1.0 
        self.cell.thermal_conductivity                    = 32.2    # [J/kgK]  
        
        
        self.cell.charging_SOC_cutoff                     = 1.
        self.cell.charging_voltage                        = 340.
        self.cell.charging_current                        = 200.
        
        self.pack_config.series                            = 1
        self.pack_config.parallel                          = 1   
        self.module_config.normal_count                    = 1       # number of cells normal to flow
        self.module_config.parallel_count                  = 1       # number of cells parallel to flow      
        self.module_config.normal_spacing                  = 0.02
        self.module_config.parallel_spacing                = 0.02
                                                           
        self.cooling_fluid.tag                             = 'air'
        self.cooling_fluid.thermal_conductivity            = 0.0253 # W/mK
        self.cooling_fluid.specific_heat_capacity          = 1006   # K/kgK
        self.cooling_fluid.discharge_air_cooling_flowspeed = 0.05   
        self.cooling_fluid.charge_air_cooling_flowspeed    = 0.05   
        self.cooling_fluid.prandlt_number_fit              = prandlt_number_model()
        
        return 

def prandlt_number_model():
    raw_Pr = np.array([[-173.2,0.780 ], [-153.2,0.759 ], [-133.2,0.747 ], 
                       [-93.2,0.731  ], [-73.2,0.726  ], [-53.2,0.721  ], 
                       [-33.2,0.717  ], [-13.2,0.713  ], [0.0,0.711    ], 
                       [6.9,0.710    ],[15.6,0.709   ], [26.9,0.707   ],
                       [46.9,0.705   ], [66.9,0.703   ], [86.9,0.701   ],
                       [106.9,0.700  ], [126.9,0.699  ], [226.9,0.698  ], 
                       [326.9,0.703  ], [426.9,0.710  ], [526.9,0.717  ],
                       [626.9,0.724  ], [  726.9,0.730 ],[826.9,0.734],
                       [1226.9,0.743], [1626.9,0.742]]) 
   
    z1  = np.polyfit(raw_Pr[:,0],raw_Pr[:,1],4)
    pnf = np.poly1d(z1)  
    
    return pnf  