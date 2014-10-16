#Battery.py
# 
# Created:  Emilio Botero, Jun 2014
# Modified:  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
import scipy as sp
from SUAVE.Attributes import Units
from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Battery Class
# ----------------------------------------------------------------------    

class Battery(Energy_Component):
    
    def __defaults__(self):
        
        self.type = 'Li-Ion'
        self.mass_properties.mass = 0.0
        self.current_energy = 0.0
        self.resistance = 0.07446
        self.specific_energy=0
        self.specific_power=0.
        self.max_energy=0.
        
        if self.type=='Li-S':
            self.specific_energy=500.*Units.Wh/Units.kg
            self.specific_power=1*Units.kW/Units.kg
        elif self.type=='Li-ion':
            self.specific_energy=90*Units.Wh/Units.kg
            self.specific_power=1*Units.kW/Units.kg
    def initialize_mass(self, energy, power):
        #initializes properties of battery based on specific energy and specific power            
        if self.specific_energy==0:
            print 'battery specific energy not specified!'
         
        if self.specific_power==0:
            print 'battery specific power not specified!'
        self.mass_properties.mass= max((energy/(self.specific_energy), self.power/(specific_power)))
        self.max_energy=self.mass_properties.mass*self.specific_energy #convert total energy to Joules
        self.max_power= self.mass_properties.mass*self.specific_power #convert total power to Watts
    
    def initialize_energy(self,mass):
        self.max_energy=mass/self.specific_energy
        if self.specific_energy==0.:
            print 'Warning, specific energy not initialized'
    
    def ragone_optimum(self, energy, power):
        '''
        For Li-S and Li-Ion batteries, determines optimum spot on ragone plot
        to size the battery
        
        Inputs:    
            energy= energy battery is required to hold [W]
            power= power battery is required to provide [W]

       Reads:
            energy
            power

       Outputs:
            battery.specific_energy
            battery.specific_power
        '''
        if self.type=='Li-S':
            esp=np.linspace(300,700,500)             #create vector of specific energy (W-h/kg)
            psp=245.848*np.power(10,-.00478*esp)     #create vector of specific power based on Ragone plot fit curve (kW/kg)
        elif self.type=='Li-Ion':
            esp=np.linspace(50,200,500)        
            psp=88.818*np.power(10,-.01533*esp)
        
        esp=esp*Units.Wh/Units.kg                   #convert specific energy to Joules/kg
        psp=psp*Units.kW/Units.kg                   #convert specific power to W/kg
        mass_energy=np.divide(energy, esp)          #vector of battery masses for mission based on energy requirements
        mass_power=np.divide(power, psp)      #vector of battery masses for mission based on power requirements
        for j in range(len(mass_energy)-1):
            mass_req.append(max(mass_energy[j],mass_power[j])) #required mass at each of the battery design points
        mass=min(mass_req)                          #choose the minimum battery mass that satisfies the mission requirements
        ibat=np.argmin(mass_req)                    #find index for minimum mass
        ebat=esp[ibat]*mass                         #total energy in the battery in J 
        
        #output values to Battery component

        self.specific_power=psp[ibat]
        self.specific_energy=esp[ibat]
        
        return
              
    
    def energy_calc(self,numerics):
        
        # Unpack
        Ibat  = self.inputs.batlogic.Ibat
        pbat  = self.inputs.batlogic.pbat
        edraw = self.inputs.batlogic.e
        Rbat  = self.resistance
        I     = numerics.integrate_time
        
        # Maximum energy
        max_energy = self.max_energy
        
        #state of charge of the battery

        x = np.divide(self.current_energy,self.max_energy())[:,0,None]

        # C rate from 
        C = 3600.*pbat/self.max_energy()
        
        # Empirical value for discharge
        x[x<-35.] = -35. # Fix x so it doesn't warn
        
        f = 1-np.exp(-20*x)-np.exp(-20*(1-x)) 
        
        f[x<0.0] = 0.0 # Negative f's don't make sense
        
        # Model discharge characteristics based on changing resistance
        R = Rbat*(1+C*f)
        
        # Calculate resistive losses
        Ploss = (Ibat**2)*R
        
        # Energy loss from power draw
        eloss = np.dot(I,Ploss)
        

        # Pack up
        self.current_energy=self.current_energy[0]*np.ones_like(eloss)
   

        delta = 0.0
        flag  = 0
        self.current_energy = self.current_energy[0] * np.ones_like(eloss) 
        for ii in range(1,len(edraw)):
            if (edraw[ii,0] > (max_energy- self.current_energy[ii-1])):
                flag = 1 
                delta = delta + ((max_energy- self.current_energy[ii-1]) - edraw[ii,0] + np.abs(eloss[ii]))
                edraw[ii,0] = edraw[ii,0] + delta
            elif flag ==1:
                edraw[ii,0] = edraw[ii,0] + delta
            self.current_energy[ii] = self.current_energy[ii] + edraw[ii] - np.abs(eloss[ii])
                    
        return  
