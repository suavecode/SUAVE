## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
# Lithium_Ion_LiNiMnCoO2_18650.py
# 
# Created:  Feb 2020, M. Clarke
# Modified: Sep 2021, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
import SUAVE
from   SUAVE.Core import Units , Data 
import numpy as np
from   scipy.interpolate import RegularGridInterpolator 
from   SUAVE.Methods.Power.Battery.Discharge_Models.LiNiMnCoO2_discharge  import LiNiMnCoO2_discharge
from   SUAVE.Methods.Power.Battery.Charge_Models.LiNiMnCoO2_charge        import LiNiMnCoO2_charge

## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
def Lithium_Ion_LiNiMnCoO2_18650(self):
    """ Specifies discharge/specific energy characteristics specific 
        18650 lithium-nickel-manganese-cobalt-oxide battery cells     
        
        Assumptions:
        Convective Thermal Conductivity Coefficient corresponds to forced
        air cooling in 35 m/s air 
        
        Source:
        Automotive Industrial Systems Company of Panasonic Group, Technical Information of 
        NCR18650G, URL https://www.imrbatteries.com/content/panasonic_ncr18650g.pdf
        
        convective  heat transfer coefficient, h 
        Jeon, Dong Hyup, and Seung Man Baek. "Thermal modeling of cylindrical 
        lithium ion battery during discharge cycle." Energy Conversion and Management
        52.8-9 (2011): 2973-2981.
        
        thermal conductivity, k 
        Yang, Shuting, et al. "A Review of Lithium-Ion Battery Thermal Management 
        System Strategies and the Evaluate Criteria." Int. J. Electrochem. Sci 14
        (2019): 6077-6107.
        
        specific heat capacity, Cp
        (axial and radial)
        Yang, Shuting, et al. "A Review of Lithium-Ion Battery Thermal Management 
        System Strategies and the Evaluate Criteria." Int. J. Electrochem. Sci 14
        (2019): 6077-6107.
        
        # Electrode Area
        Muenzel, Valentin, et al. "A comparative testing study of commercial
        18650-format lithium-ion battery cells." Journal of The Electrochemical
        Society 162.8 (2015): A1592.
        
        Inputs:
        None
        
        Outputs:
        None
        
        Properties Used:
        N/A
    """       
    self.tag                              = 'Lithium_Ion_LiNiMnCoO2_Cell' 
                                          
    self.cell.diameter                    = 0.018   # [m]
    self.cell.height                      = 0.06485 # [m]
    self.cell.surface_area                = (np.pi*self.cell.height*self.cell.diameter) + (0.5*np.pi*self.cell.diameter**2)   # [m^2]
    self.cell.density                     = 1760    # [kg/m^3] 
    self.cell.volume                      = np.pi*(0.5*self.cell.diameter)**2*self.cell.height  # [m^3] 
    self.cell.mass                        = 0.048 * Units.kg 
    self.cell.electrode_area              = 0.0342  # [m^2] 
                                          
    self.cell.max_voltage                 = 4.2     # [V]
    self.cell.nominal_capacity            = 3.55    # [Amp-Hrs]
    self.cell.nominal_voltage             = 3.6     # [V] 
    self.cell.charging_voltage            = self.cell.nominal_voltage # [V] 
    
    self.watt_hour_rating                 = self.cell.nominal_capacity  * self.cell.nominal_voltage  # [Watt-hours]      
    self.specific_energy                  = self.watt_hour_rating*Units.Wh/self.cell.mass            # [J/kg]
    self.specific_power                   = self.specific_energy/self.cell.nominal_capacity          # [W/kg]   
    self.resistance                       = 0.025   # [Ohms]
                                                   
    self.specific_heat_capacity           = 1108    # [J/kgK]  
    self.cell.specific_heat_capacity      = 1108    # [J/kgK]    
    self.cell.radial_thermal_conductivity = 0.4     # [J/kgK]  
    self.cell.axial_thermal_conductivity  = 32.2    # [J/kgK] # estimated   
    
    self.discharge_model                  = LiNiMnCoO2_discharge
    self.charge_model                     = LiNiMnCoO2_charge 
                                          
    battery_raw_data                      = load_battery_results()                                                   
    self.discharge_performance_map        = create_discharge_performance_map(battery_raw_data)  
    
    return  

def create_discharge_performance_map(battery_raw_data):
    """ Create discharge and charge response surface for 
        LiNiMnCoO2 battery cells 
        
        Source:
        N/A
        
        Assumptions:
        N/A
        
        Inputs: 
            
        Outputs: 
        battery_data

        Properties Used:
        N/A
                                
    """  
    
    # Process raw data 
    processed_data = process_raw_data(battery_raw_data)
    
    # Create performance maps 
    battery_data = create_response_surface(processed_data) 
    
    return battery_data

def create_response_surface(processed_data):
    
    battery_map             = Data() 
    amps                    = np.linspace(0, 8, 5)
    temp                    = np.linspace(0, 50, 6) +  272.65
    SOC                     = np.linspace(0, 1, 15) 
    battery_map.Voltage     = RegularGridInterpolator((amps, temp, SOC), processed_data.Voltage)
    battery_map.Temperature = RegularGridInterpolator((amps, temp, SOC), processed_data.Temperature)
     
    return battery_map 

def process_raw_data(raw_data):
    """ Takes raw data and formats voltage as a function of SOC, current and temperature
        
        Source 
        N/A
        
        Assumptions:
        N/A
        
        Inputs:
        raw_Data     
            
        Outputs: 
        procesed_data 

        Properties Used:
        N/A
                                
    """
    processed_data = Data()
     
    processed_data.Voltage        = np.zeros((5,6,15,2)) # current , operating temperature , SOC vs voltage      
    processed_data.Temperature    = np.zeros((5,6,15,2)) # current , operating temperature , SOC  vs temperature 
    
    # Reshape  Data          
    raw_data.Voltage 
    for i, Amps in enumerate(raw_data.Voltage):
        for j , Deg in enumerate(Amps):
            min_x    = 0 
            max_x    = max(Deg[:,0])
            x        = np.linspace(min_x,max_x,15)
            y        = np.interp(x,Deg[:,0],Deg[:,1])
            vec      = np.zeros((15,2))
            vec[:,0] = x/max_x
            vec[:,1] = y
            processed_data.Voltage[i,j,:,:]= vec   
            
    for i, Amps in enumerate(raw_data.Temperature):
        for j , Deg in enumerate(Amps):
            min_x    = 0   
            max_x    = max(Deg[:,0])
            x        = np.linspace(min_x,max_x,15)
            y        = np.interp(x,Deg[:,0],Deg[:,1])
            vec      = np.zeros((15,2))
            vec[:,0] = x/max_x
            vec[:,1] = y
            processed_data.Temperature[i,j,:,:]= vec     
    
    return  processed_data  

def load_battery_results():
    return SUAVE.Input_Output.SUAVE.load('NMC_Raw_Data.res')
