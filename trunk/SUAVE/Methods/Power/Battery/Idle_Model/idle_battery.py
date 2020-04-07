## @ingroup Methods-Power-Battery-Idle_Model
# idle_battery.py
# 
# Created: Apr 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data , Units 
import numpy as np
from scipy.integrate import  cumtrapz 

def idle_battery(battery,numerics): 
    """This is an idle model for all 18650 lithium-ion batteries. In this thermodynamic 
       model, the heat energy generated in a battery during charge or discharge is allowed 
       to dissipate in the atmosphere  
       
       Source: 
       Cell Charge: Chin, J. C., Schnulo, S. L., Miller, T. B., Prokopius, K., and Gray, 
       J., “"Battery Performance Modeling on Maxwell X-57",”AIAA Scitech, San Diego, CA,
       2019. URLhttp://openmdao.org/pubs/chin_battery_performance_x57_2019.pdf.     
       
       Cell Heat Coefficient:  Wu et. al. "Determination of the optimum heat transfer 
       coefficient and temperature rise analysis for a lithium-ion battery under 
       the conditions of Harbin city bus driving cycles". Energies, 10(11). 
       https://doi.org/10.3390/en10111723
       
       Inputs:
         battery. 
               cell_mass         (battery cell mass)                   [kilograms]
               Cp                (battery cell specific heat capacity) [J/(K kg)]
               h                 (heat transfer coefficient)           [W/(m^2*K)]
               t                 (battery age in days)                 [days]
               cell_surface_area (battery cell surface area)           [meters^2]
               T_ambient         (ambient temperature)                 [Degrees Celcius]
               T_current         (pack temperature)                    [Degrees Celcius]
               T_cell            (battery cell temperature)            [Degrees Celcius]
       
       Outputs:
         battery.          
              cell_temperature                                         [Degrees Celcius]
        
    """
    
    # Unpack varibles   
    cell_mass         = battery.cell.mass            
    Cp                = battery.cell.specific_heat_capacity   
    t                 = battery.age_in_days
    cell_surface_area = battery.cell.surface_area
    T_ambient         = battery.ambient_temperature    
    T_current         = battery.temperature      
    T_cell            = battery.cell_temperature     
    I                 = numerics.time.integrate
    D                 = numerics.time.differentiate    
    
    # Determine temperature increase
    h = -290 + 39.036*T_current[0] - 1.725*(T_current[0]**2) + 0.026*(T_current[0]**3)    
    P_net      = h*0.5*cell_surface_area*(T_current - T_ambient)
    dT_dt      = P_net/(cell_mass*Cp)
    T_current  = T_current[0] - np.dot(I,dT_dt)  
     
    # Pack outputs
    battery.cell_temperature         = T_current   
    
    return battery
    
