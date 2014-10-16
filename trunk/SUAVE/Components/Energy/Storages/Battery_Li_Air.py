#Battery.py
# 
# Created:  Michael Vegh
# Modified: October, 2014

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

class Battery_Li_Air(Energy_Component):
    
    def __defaults__(self):
        
        self.type = 'Li-Air'
        self.mass_properties.mass = 0.0
        self.current_energy = 0.0
        self.resistance = 0.07446
        self.specific_energy=2000.*Units.Wh/Units.kg    #convert to Joules/kg
        self.specific_power=0.66*Units.kW/Units.kg      #convert to W/kg
        self.max_energy=0.
        self.max_power=0.
    def initialize(self, energy, power): 
        #initializes properties of battery based on specific energy and specific power            
        if self.specific_energy==0:
            print 'battery specific energy not specified!'
         
        if self.specific_power==0:
            print 'battery specific power not specified!'
        
        self.mass_properties.mass= max(energy/self.specific_energy, power/self.specific_power)
        self.max_energy=self.mass_properties.mass*self.specific_energy #convert total energy to Joules
        self.max_power= self.mass_properties.mass*self.specific_power #convert total power to Watts
        self.current_energy=self.max_energy
 
    
    def energy_calc(self,numerics):
        
        # Unpack
        Ibat  = self.inputs.batlogic.Ibat
        pbat  = self.inputs.batlogic.pbat
        edraw = self.inputs.batlogic.e
        Rbat  = self.resistance
        I     = numerics.integrate_time
       
        # X value
        x = np.divide(self.current_energy,self.max_energy)[:,0,None]
        
        # C rate from 
        C = 3600.*pbat/self.max_energy
        
        # Empirical value for discharge
        x[x<-35.] = -35. # Fix x so it doesn't warn
        
        f = 1-np.exp(-20*x)-np.exp(-20*(1-x)) 
        
        f[x<0.0] = 0.0 # Negative f's don't make sense
        
        # Model discharge characteristics based on changing resistance
        R = Rbat*(1+C*f)
        
        #Calculate resistive losses
        Ploss = -(Ibat**2)*R
        
        # Energy loss from power draw
        eloss = np.dot(I,Ploss)
        
        # Pack up
        '''
        self.current_energy = self.current_energy[0] + edraw + eloss
        self.current_energy[self.current_energy>self.max_energy] = self.max_energy
        '''
        
        self.current_energy=self.current_energy[0]*np.ones_like(eloss)
   

        delta = 0.0
        flag  = 0
        for ii in range(1,len(edraw)):
            if (edraw[ii,0] > (self.max_energy- self.current_energy[ii-1])):
                flag = 1 
                delta = delta + ((self.max_energy- self.current_energy[ii-1]) - edraw[ii,0] + np.abs(eloss[ii]))
                edraw[ii,0] = edraw[ii,0] + delta
            elif flag ==1:
                edraw[ii,0] = edraw[ii,0] + delta
            self.current_energy[ii] = self.current_energy[ii] + edraw[ii] - np.abs(eloss[ii])
        
        
      
        mdot=(pbat+Ploss) *(1.92E-4)/Units.Wh  #weight gain of battery (positive means mass loss)
        return  mdot
    def find_mass_gain(self):               #find total potential mass gain of a lithium air battery
        mgain=self.max_energy*(1.92E-4)/Units.Wh
        
        return mgain   
            
  