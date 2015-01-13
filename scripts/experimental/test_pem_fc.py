
import sys
sys.path.append('../trunk')
import SUAVE
from SUAVE.Components.Energy.Converters.PEM_FC import PEM_FC
import numpy as np

def main():
    N=4
    thermo=SUAVE.Core.Data()
    air=SUAVE.Attributes.Gases.Air()

    thermo.Tt=np.ones(N)*300.
    thermo.pt=np.ones(N)*100E3
    cp=air.compute_cp(thermo.Tt[0],thermo.pt[0])
    thermo.cp=cp*np.ones(N)
    
    gamma=air.compute_gamma(thermo.Tt[0],thermo.pt[0])
    thermo.gamma=gamma*np.ones(N)
    thermo.ht=np.ones(N)*thermo.cp[0]*thermo.Tt[0]
    maxpower=10000.
    aircraft    = SUAVE.Vehicle()
    compressor=SUAVE.Components.Propulsors.Segments.Compressor()
    compressor.eta_polytropic=[.6]
    compressor.pt_ratio=(2*101.3*10**3)/thermo.pt[0]
    
    mdot=.01;            #assign an overall mass flow rate
    fuel_cell=SUAVE.Components.Energy.Converters.PEM_FC()
  
    fuel_cell.active=1
    compressor.active=1
  
    SUAVE.Methods.Power.size_pem_fc(fuel_cell,maxpower)
    
    power=maxpower/2
    compressor.i=0
    compressor.f=1
    fuel_cell.i=1
    fuel_cell.f=2
    compressor(thermo,power)
    power=power+mdot*(thermo.ht[1]-thermo.ht[0])

    [mdot_h2,mdot_products]=fuel_cell(power,thermo, mdot)
    print mdot_products
    print thermo
    #now run fuel cell without specifying any thermodynamic properties
    [mdot_h2,mdot_products]=fuel_cell(power)
if __name__ == '__main__':
    main()