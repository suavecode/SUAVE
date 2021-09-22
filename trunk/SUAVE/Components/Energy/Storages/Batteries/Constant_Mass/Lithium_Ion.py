
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
import SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion_LiNCA_18650 as Lithium_Ion_LiNCA_18650
import SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion_LiNiMnCoO2_18650 as Lithium_Ion_LiNiMnCoO2_18650
import SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion_LiFePO4_18650 as Lithium_Ion_LiFePO4_18650

# ----------------------------------------------------------------------
#  Lithium_Ion
# ----------------------------------------------------------------------    
## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
class Lithium_Ion(Battery):
    """
    Specifies discharge/specific energy characteristics specific tobytes
    lithium-ion batteries 
       
    Assumptions:
    Convective Thermal Conductivity Coefficient corresponds to forced
    air cooling in 35 m/s air 
    
    Source:
    Convective Heat Transfer Coefficient:  
    Wu et. al. "Determination of the optimum heat transfer 
    coefficient and temperature rise analysis for a lithium-ion battery under 
    the conditions of Harbin city bus driving cycles". Energies, 10(11). 
    https://doi.org/10.3390/en10111723
    
    Inputs:
    None
    
    Outputs:
    None
    
    Properties Used:
    N/A
    """     
    def __defaults__(self,battery_chemistry='LFP'):
            
        self.tag               = 'Generic_Lithium_Ion_Battery_Cell'         
        self.cell_chemistry    = battery_chemistry
        self.cell              = Data()   
        self.module            = Data()        
        self.pack_config       = Data()
        self.module_config     = Data()
        self.cooling_fluid     = Data()
        
        self.cell.charging_SOC_cutoff                      = 1. 
        self.cell.charging_current                         = 3.0     # [Amps]
                         
        self.convective_heat_transfer_coefficient          = 35.     # [W/m^2K] 
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
        
        if battery_chemistry == 'NCA':
            self = self.Lithium_Ion_LiNCA_18650()     
        elif battery_chemistry == 'NMC': 
            self = self.Lithium_Ion_LiNiMnCoO2_18650() 
        elif battery_chemistry == 'LFP': 
            self = self.Lithium_Ion_LiFePO4_18650()  