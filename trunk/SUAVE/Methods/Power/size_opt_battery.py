"""sizes an optimized battery based on power and energy requirements based on a Ragone plot curve fit"""
#by M. Vegh

""" SUAVE Methods for Energy Systems """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

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


    if battery.type=="Li_S":
        esp=np.linspace(300,700,500)         #create vector of specific energy (W-h/kg)
        psp=245.848*np.power(10,-.00478*esp) #create vector of specific power based on Ragone plot fit curve (kW/kg)
        rho=1050.                          #[kg/m^3]
    if battery.type=="Li_ion":
        esp=np.linspace(50,200,500)          #create vector of specific energy (W-h/kg)
        psp=88.818*np.power(10,-.01533*Esp)  #create vector of specific power based on Ragone plot fit curve (kW/kg)
        rho=2000.                          # [kg/m^3]

    mass_req=[]
    esp=esp*3600                             #convert specific energy to Joules/kg
    mass_energy=np.divide(energy, esp)         #vector of battery masses for mission based on energy requirements
    mass_power=np.divide(power/1000., psp)    #vector of battery masses for mission based on power requirements
    for j in range(len(mass_energy)-1):
        mass_req.append(max(mass_energy[j],mass_power[j])) #required mass at each of the battery design points
    mass=min(mass_req)                       #choose the minimum battery mass that satisfies the mission requirements
    ibat=np.argmin(mass_req)                 #find index for minimum mass
    ebat=esp[ibat]*mass                      #total energy in the battery in J
    
    volume=mass/rho                          #total volume of the battery
    esp=esp/3600                             #convert back to W-h/kg

    #output values to Battery component
    battery.Mass_Props.mass=mass
    battery.Volume=volume
    battery.TotalEnergy=ebat
    battery.CurrentEnergy=ebat
    battery.SpecificPower=psp[ibat]
    battery.SpecificEnergy=esp[ibat]
    battery.MaxPower=mass*1000*psp[ibat]
    battery.MassDensity=mass/volume

    return