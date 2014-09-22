"""Battery_Li_S.Py: calculates battery discharge losses when run """
# M. Vegh
#
"""Sources are the "Requirements for a Hydrogen Powered All-Electric
Manned Helicopter" Anubhav Datta and Wayne Johnson, "Developing Li-S Chemistry for
 High-Energy Rechargable Batteries," by John Affinito
"""
# ------------------------------------------------------------
#  Imports 
# ------------------------------------------------------------

from Storage import Storage
from Battery import Battery
import numpy as np
# ------------------------------------------------------------
#  Battery
# ------------------------------------------------------------
    
class Battery_Li_Ion(Battery):
    """ SUAVE.Attributes.Components.Energy.Storage.Battery()
    """
    def __defaults__(self):
        self.MassDensity = 2000.      # kg/m^3
        self.SpecificEnergy = 0.0     # W-hr/kg
        self.SpecificPower = 0.0      # kW/kg
        self.MaxPower=0.0             # W
        self.TotalEnergy = 0.0        # J
        self.CurrentEnergy=0.0        # J
        self.Volume = 0.0             # m^3
        self.R0=.07446                #base resistance (ohms)
        
    def find_opt_mass(self, energy, power):
        """Inputs:
            
            energy= energy battery is required to hold [W]
            power= power battery is required to provide [W]

       Reads:
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
        esp=np.linspace(50,200,500)              #create vector of specific energy (W-h/kg)
        psp=88.818*np.power(10,-.01533*esp)      #create vector of specific power based on Ragone plot fit curve (kW/kg)
        mass_req=[]
        esp=esp*3600                             #convert specific energy to Joules/kg
        mass_energy=np.divide(energy, esp)       #vector of battery masses for mission based on energy requirements
        mass_power=np.divide(power/1000., psp)   #vector of battery masses for mission based on power requirements
        for j in range(len(mass_energy)-1):
            mass_req.append(max(mass_energy[j],mass_power[j])) #required mass at each of the battery design points
        mass=min(mass_req)                       #choose the minimum battery mass that satisfies the mission requirements
        ibat=np.argmin(mass_req)                 #find index for minimum mass
        ebat=esp[ibat]*mass                      #total energy in the battery in J 
        volume=mass/self.MassDensity             #total volume of the battery
        esp=esp/3600                             #convert back to W-h/kg
        #output values to Battery component
        self.Mass_Props.mass=mass
        self.Volume=volume
        self.TotalEnergy=ebat
        self.CurrentEnergy=ebat
        self.SpecificPower=psp[ibat]
        self.SpecificEnergy=esp[ibat]
        self.MaxPower=mass*1000*psp[ibat]
        return