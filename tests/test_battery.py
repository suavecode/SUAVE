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
    battery     = SUAVE.Components.Energy.Storages.Battery({"tag":"Battery"})
    SUAVE.Methods.Power.size_opt_battery(battery,Ereq, Preq)
    

    print battery
    time=60; #time in seconds
    #run the battery
    Ecurrent=battery.TotalEnergy
    Ploss=battery(Preq,time)
    print battery
if __name__ == '__main__':
    main()