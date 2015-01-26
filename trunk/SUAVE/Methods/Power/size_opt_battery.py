"""sizes an optimized battery based on power and energy requirements based on a Ragone plot curve fit"""
#by M. Vegh

""" SUAVE Methods for Energy Systems """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import scipy
# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def size_opt_battery(battery, energy, power): #adds a battery that is optimized based on power and energy requirements and technology
    """Inputs:
            battery
            energy= energy battery is required to hold [W]
            power= power battery is required to provide [W]

       Reads:
            battery.type
            energy
            power

       Outputs:
            battery.Mass_Props.mass=battery mass [kg]                                                                                                                                                                                                                                            
            battery.Volume=battery volume [m^3]                                                                                                                                                                                       
            battery.TotalEnergy=total energy in the battery [J]                                                                                                                                                                                                        
            battery.SpecificPower=specific power of the battery [kW/kg]                                                                                                                                                                            
            battery.SpecificEnergy=specific energy of the battery [W-h/kg]                                                                                                                                                                              
            battery.MassDensity=battery volume [kg/m^3]
    """

    '''
    if battery.type=="Li_S":
        #esp=np.linspace(300,700,500)         #create vector of specific energy (W-h/kg)
        psp=245.848*np.power(10,-.00478*esp) #create vector of specific power based on Ragone plot fit curve (kW/kg)
        rho=1050.                          #[kg/m^3]
    if battery.type=="Li_ion":
        #esp=np.linspace(50,200,500)          #create vector of specific energy (W-h/kg)
        psp=88.818*np.power(10,-.01533*Esp)  #create vector of specific power based on Ragone plot fit curve (kW/kg)
        rho=2000.                          # [kg/m^3]

    '''
    
    lb=battery.ragone.lower_bound
    ub=battery.ragone.upper_bound
    return