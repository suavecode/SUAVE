## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
# Lithium_Ion_LiNCA_18650.py
# 
# Created:  Feb 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
from SUAVE.Core import Units , Data 
from scipy.interpolate import  RectBivariateSpline
import numpy as np

from SUAVE.Components.Energy.Storages.Batteries                     import Battery 
from SUAVE.Methods.Power.Battery.Discharge_Models.LiNCA_discharge   import LiNCA_discharge 
from SUAVE.Methods.Power.Battery.Charge_Models.LiNCA_charge         import LiNCA_charge 

## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
class Lithium_Ion_LiNCA_18650(Battery): 
    """ Specifies discharge/specific energy characteristics specific 
        18650 lithium-nickel-cobalt-aluminum oxide (LiNCA) battery cells   
        
        Assumptions:
        Convective Thermal Conductivity Coefficient corresponds to forced
        air cooling in 35 m/s air 
        
        Source:
        Intriduction of INR18650-30Q. https://eu.nkon.nl/sk/k/30q.pdf
        
        convective  heat transfer coefficient, h 
        Jeon, Dong Hyup, and Seung Man Baek. "Thermal modeling of cylindrical 
        lithium ion battery during discharge cycle." Energy Conversion and Management
        52.8-9 (2011): 2973-2981.
        
        thermal conductivity, k 
        Yang, Shuting, et al. "A Review of Lithium-Ion Battery Thermal Management 
        System Strategies and the Evaluate Criteria." Int. J. Electrochem. Sci 14
        (2019): 6077-6107.
        
        specific heat capacity, Cp
        Yang, Shuting, et al. "A Review of Lithium-Ion Battery Thermal Management 
        System Strategies and the Evaluate Criteria." Int. J. Electrochem. Sci 14
        (2019): 6077-6107. 
        
        Inputs:
        None
        
        Outputs:
        None
        
        Properties Used:
        N/A
    """  
    
    def __defaults__(self):
        self.tag                                          = 'Lithium_Ion_Battery_Cell' 
        self.cell                                         = Data()   
        self.module                                       = Data()        
        self.pack_config                                  = Data()
        self.module_config                                = Data()
        self.cooling_fluid                                = Data()
                                              
        self.cell.mass                                    = 0.048 * Units.kg 
        self.cell.diameter                                = 0.01833  # [m]
        self.cell.height                                  = 0.06485  # [m] 
        self.cell.surface_area                            = (np.pi*self.cell.height*self.cell.diameter) + (0.5*np.pi*self.cell.diameter**2)  # [m^2]
        self.cell.density                                 = 760       # [kg/m^3] 
        self.cell.volume                                  = 3.2E-5    # [m^3] 
        self.cell.electrode_area                          = 0.0342    # [m^2] 
                                                          
        self.cell.max_voltage                             = 4.2   # [V]
        self.cell.nominal_capacity                        = 3.00  # [Amp-Hrs]
        self.cell.nominal_voltage                         = 3.6   # [V]
        self.cell.charging_SOC_cutoff                     = 1.         
        self.cell.charging_voltage                        = self.cell.nominal_voltage   # [V]  
        self.cell.charging_current                        = 3.0    
        self.watt_hour_rating                             = self.cell.nominal_capacity  * self.cell.nominal_voltage  # [Watt-hours]      
        self.specific_energy                              = self.watt_hour_rating*Units.Wh/self.cell.mass            # [J/kg]
        self.specific_power                               = self.specific_energy/self.cell.nominal_capacity          # [W/kg]   
        self.resistance                                   = 0.025   # [Ohms]
                                                          
        self.specific_heat_capacity                       = 837.4   # [J/kgK] 
        self.cell.specific_heat_capacity                  = 837.4   # [J/kgK]  
        self.heat_transfer_coefficient                    = 35      # [W/m^2K]       
        self.heat_transfer_efficiency                     = 1.0 
        self.cell.thermal_conductivity                    = 32.2    # [J/kgK]   
                                                         
        self.pack_config.series                           = 1
        self.pack_config.parallel                         = 1   
        self.pack_config.total                            = 1   
        self.module_config.total                          = 1  
        self.module_config.normal_count                   = 1    # number of cells normal to flow
        self.module_config.parallel_count                 = 1    # number of cells parallel to flow      
        self.module_config.normal_spacing                 = 0.02
        self.module_config.parallel_spacing               = 0.02

                                                   
        self.cooling_fluid.tag                             = 'air'
        self.cooling_fluid.thermal_conductivity            = 0.0253 # W/mK
        self.cooling_fluid.specific_heat_capacity          = 1006   # K/kgK
        self.cooling_fluid.discharge_air_cooling_flowspeed = 0.01   
        self.cooling_fluid.charge_air_cooling_flowspeed    = 0.01     
           
        self.discharge_model                               = LiNCA_discharge 
        self.charge_model                                  = LiNCA_charge 
                                                           
        self.discharge_performance_map                     = create_discharge_performance_map()        
        return  
    
def create_discharge_performance_map():
    """ Create discharge and charge response surface for 
        LiNCA  battery cells     
        
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
    battery_data = Data()
    T_bp = np.array([0., 20., 30., 45.]) + 272.65
    SOC_bp = np.array( [0. , 0.03333333, 0.06666667, 0.1 , 0.13333333, 0.16666667,
                        0.2 , 0.23333333, 0.26666667, 0.3 , 0.33333333, 0.36666667,
                        0.4 , 0.43333333, 0.46666667, 0.5 , 0.53333333, 0.56666667,
                        0.6 , 0.63333333, 0.66666667, 0.7 , 0.73333333, 0.76666667,
                        0.8 , 0.83333333, 0.86666667, 0.9 , 0.93333333, 0.96666667,1.] )
     
    tV_oc = np.array([ [2.92334783,3.00653623,3.08972464,3.17291304,3.23989855,
                        3.31010145, 3.3803913 ,3.44033333,3.49033333,3.52169565,
                        3.54391304,3.58695652, 3.62095652,3.65437681, 3.68604348,
                        3.72430435,3.75531884,3.79102899, 3.82030435,3.84181159,
                        3.86124638, 3.88921739,3.91686957,3.96223188, 4.00169565,
                        4.04117391,4.06849275,4.07573913, 4.08571014,4.10571014,
                        4.161 ] , 
                       [2.99293893,3.05400763,3.11507634,3.17614504, 3.23506616,
                        3.30371247, 3.37521374, 3.43605852,3.48697455,3.5200229 ,
                        3.54251908,3.58374046, 3.6329313 ,3.67379644,3.70287532,
                        3.73784733,3.76526463,3.79174809, 3.81922901,3.84108142,
                        3.87212214, 3.90738931,3.93615267,3.98113995, 4.02093893,
                        4.04504071,4.07114758,4.07583969, 4.08371501,4.10560814, 
                        4.161 ] , 
                       [2.84084639,2.98428484,3.1050295 ,3.19464496,3.25566531,
                        3.309059 , 3.37185148, 3.43473652,3.49059613,3.51955239,
                        3.541353 ,3.58558494, 3.62641607,3.6708881 ,3.70814547,
                        3.7392177 ,3.76822075,3.79592981, 3.82260427,3.84986368,
                        3.88146592,  3.91739674,3.94798779,3.98188403, 4.02274568,
                        4.05623296,4.06830824,4.07468871, 4.08175788,4.10853306, 
                        4.153 ] ,
                       [2.81925101,2.97410931,3.09861134,3.18674899,3.24142105,
                        3.29678138, 3.35963563, 3.42195951,3.47637247,3.51383806,
                        3.54319838,3.59076923, 3.61940891,3.65574089, 3.7067004 ,
                        3.74153441,3.77023887,3.79773684, 3.82421053,3.85139271,
                        3.88311336, 3.91906478,3.94918219,3.98310931, 4.02401215,
                        4.05611741,4.07036842,4.07774494, 4.08190283,4.10867206,
                        4.153 ] ])
    
    tC_Th = np.array([ [2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,
                        2000.,2000.,2000., 2000.,2000.,2000.,2000.,2000.,2000.,
                        2000.,2000.,2000.,2000.,2000.,2000., 2000.,2000.,
                        2000.,2000.,2000.,2000.,2000.] ,
                       [2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,
                        2000.,2000.,2000., 2000.,2000.,2000.,2000.,2000.,2000.,
                        2000.,2000.,2000.,2000.,2000.,2000., 2000.,2000.,
                        2000.,2000.,2000.,2000.,2000.] ,
                       [2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,
                        2000.,2000.,2000., 2000.,2000.,2000.,2000.,2000.,2000.,
                        2000.,2000.,2000.,2000.,2000.,2000., 2000.,2000.,
                        2000.,2000.,2000.,2000.,2000.] , 
                       [2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,
                        2000.,2000.,2000., 2000., 2000.,2000.,2000.,2000.,2000.,
                        2000.,2000.,2000.,2000.,2000.,2000., 2000.,2000.,
                        2000.,2000.,2000.,2000.,2000.] ])
    
    tR_Th = np.array([ [0.09 ,0.09 ,0.09 ,0.09 ,0.07130435 ,0.06 , 0.06 ,
                        0.06 ,0.06 ,0.06 ,0.06 ,0.06 ,  0.08217391,
                        0.07492754, 0.07 ,0.07 ,0.07 ,0.07 , 0.07 ,
                        0.05318841,0.04144928, 0.0573913 ,0.06884058,
                        0.07, 0.07456522,0.075,0.075,0.05586957,0.055
                        ,0.04021739, 0.04 ] ,
                       [0.08534351,0.07516539,0.06498728,0.05480916,
                        0.04838931,0.04589059, 0.045 ,0.045 ,0.04195929,
                        0.03937405,0.03642494,0.035 , 0.035 ,0.03848601,
                        0.05430025,0.04534351,0.03624682,0.03115776, 0.03,
                        0.03 ,0.03 ,0.03839695,0.04 ,0.04 , 0.04 ,0.03089059,
                        0.03,0.03 ,0.02807125, 0.02505344 , 0.02],
                       [0.0677823 ,0.05252289,0.045 ,0.045,0.045,0.045 ,
                        0.04207528,0.04 ,0.03690234,0.035 ,0.0317294 ,
                        0.02798576,0.027 ,0.025588 ,0.025 ,0.02129705,
                        0.02   ,0.02 , 0.04377416,0.04190234,0.04,0.04 ,
                        0.04 ,0.03121058,0.02820753,0.028,0.02055341,0.02 ,
                        0.02,0.02 ,0.001 ] ,
                       [0.06728745,0.04704453,0.04,0.04 ,0.04 ,0.04 ,  0.04 ,
                        0.04  ,0.03267206,0.03 ,0.03 ,0.03 , 0.03,
                        0.02603239,0.025 ,0.02091093,0.02,0.02 , 0.04562753,
                        0.04133603,0.04  ,0.04  ,0.04 ,0.0308502 ,
                        0.02814575,0.028 ,0.02038866,0.02 ,0.02 ,0.02,  0.001 ]]) 
    
    tR_0 = np.array([ [0.2473913 ,0.20681159,0.16623188,0.12565217,0.09753623,
                       0.08362319, 0.08,0.07666667,0.0715942 ,0.07 ,0.0415942 ,
                       0.05681159, 0.067 ,0.067 ,0.067 ,0.067 ,0.067 ,0.06537681,
                       0.065 ,0.065 ,0.065 ,0.065 ,0.065 ,0.065 , 0.065 ,0.065 ,
                       0.065 ,0.065 ,0.065 , 0.065 , 0.065 ] ,
                      [0.08801527,0.07274809,0.05748092,0.04221374,0.03231552,
                       0.02722646, 0.025 ,0.025 ,0.025 ,0.025 ,0.025 ,0.025 ,
                       0.025 ,0.025 ,0.025 , 0.025 ,0.025 ,0.025 , 0.025 ,0.025 ,
                       0.025 ,0.025 ,0.025 ,0.025 , 0.025 ,0.025 ,0.025 ,0.025 ,
                       0.025 ,0.025 , 0.025 ] ,
                      [0.0677823 ,0.05252289,0.03726348,0.02733469,0.025 ,0.025 ,
                       0.025 ,0.025 ,0.025 ,0.025 ,0.025 ,0.025 , 0.025 ,0.025 , 
                       0.025 ,0.025 ,0.025 ,0.025 , 0.01311292,0.01809766,0.02 ,
                       0.02 ,0.02430824,0.025 ,  0.025 ,0.025 ,0.025 ,0.025 ,
                       0.025 ,0.025 , 0.03] , 
                      [0.06546559,0.0502834 ,0.03510121,0.02663968,0.025 ,0.025 ,
                       0.025 ,0.025 ,0.025 ,0.025 ,0.025 ,0.025 , 0.025 ,0.025 ,
                       0.025 ,0.025 ,0.025 ,0.025 , 0.01218623,0.01,0.01 ,
                       0.01890688,0.02451417,0.025 , 0.025 ,0.025 ,0.025 ,0.025 ,
                       0.025 ,0.025 , 0.03] ])
    
    SMOOTHING = 0.1 # more is more smooth, less true to the data
    battery_data.V_oc_interp = RectBivariateSpline(T_bp, SOC_bp, tV_oc, s=SMOOTHING) # % need Deg C
    battery_data.C_Th_interp = RectBivariateSpline(T_bp, SOC_bp, tC_Th, s=SMOOTHING) # % need Deg C
    battery_data.R_Th_interp = RectBivariateSpline(T_bp, SOC_bp, tR_Th, s=SMOOTHING) # % need Deg C
    battery_data.R_0_interp = RectBivariateSpline(T_bp, SOC_bp, tR_0, s=SMOOTHING)   # % need Deg C
 
    return battery_data
 