#test battery.py
import sys
sys.path.append('../trunk')
import SUAVE
from SUAVE.Components.Energy.Storages.Battery import Battery
import numpy as np
def main():
    #size the battery
    Mission_total=SUAVE.Attributes.Missions.Mission()
    Ereq=1600000. #required energy for the mission in Joules
    Preq=1000. #maximum power requirements for mission in W
   
    aircraft    = SUAVE.Vehicle()
    battery_li_air     = SUAVE.Components.Energy.Storages.Battery_Li_Air()
    battery_li_ion     = SUAVE.Components.Energy.Storages.Battery_Li_Ion()
    battery_li_s     = SUAVE.Components.Energy.Storages.Battery_Li_S()
    battery_li_s.find_opt_mass(Ereq,Preq)
    battery_li_ion.find_opt_mass(Ereq,Preq)

    print battery_li_s
    battery_li_air.Mass_Props.mass=max(Ereq*(1./3600.)/battery_li_air.SpecificEnergy, 
    (Preq/1000.)/battery_li_air.SpecificPower)
    battery_li_air.TotalEnergy=battery_li_air.Mass_Props.mass * \
    battery_li_air.SpecificEnergy*3600.
    print battery_li_air.MassDensity
    battery_li_air.Volume=battery_li_air.Mass_Props.mass/battery_li_air.MassDensity
    battery_li_air.CurrentEnergy= battery_li_air.TotalEnergy
    battery_li_air.MaxPower= (battery_li_air.SpecificPower*1000.)* \
battery_li_air.Mass_Props.mass
    time=60; #time in seconds
    #run the battery
    Ecurrent_li_s=battery_li_s.TotalEnergy
    Ploss_li_s=battery_li_s(Preq,time)
    print battery_li_s
    [Ploss_li_air, mdot]=battery_li_air(Preq,time)
    print battery_li_air
 
    Ploss_li_ion=battery_li_ion(Preq,time)
    print battery_li_ion
if __name__ == '__main__':
    main()