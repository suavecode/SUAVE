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
        self.energy_density       = 0.0
        self.current_energy       = 0.0
        self.resistance           = 0.0
        
    def max_energy(self):
        """ The maximum energy the battery can hold
            
            Inputs:
                battery mass
                battery type
               
            Outputs:
                maximum energy the battery can hold
               
            Assumptions:
                This is a simple battery, based on the model by:
                AIAA 2012-5405 by Anubhav Datta/Johnson
               
        """
        
        #These need to be fixed
        if self.type=='Li-Ion':
            if self.energy_density == 0.0:
                self.energy_density = 250.*3600.
            return self.energy_density*self.mass_properties.mass
        
        elif self.type=='Li-Po':
            if self.energy_density == 0.0:
                self.energy_density = 0.90*(10**6)           
            return self.energy_density*self.mass_properties.mass
        
        elif self.type=='Li-S':
            if self.energy_density == 0.0:
                self.energy_density = 500.*3600.            
            return self.energy_density*self.mass_properties.mass       
    
    def energy_calc(self,numerics):
        
        # Unpack
        Ibat  = self.inputs.batlogic.Ibat
        pbat  = self.inputs.batlogic.pbat
        edraw = self.inputs.batlogic.e
        Rbat  = self.resistance
        I     = numerics.integrate_time
        
        # Maximum energy
        max_energy = self.max_energy()
        
        # X value
        x = np.divide(self.current_energy,max_energy)[:,0,None]
        
        # C rate from 
        C = 3600.*pbat/self.max_energy()
        
        # Empirical value for discharge
        x[x<-34.] = -34. # Fix x so it doesn't warn
        
        f = 1-np.exp(-20*x)-np.exp(-20*(1-x)) 
        
        f[x<0.0] = 0.0 # Negative f's don't make sense
        f[np.isnan(f)] = 0.0
        
        # Model discharge characteristics based on changing resistance
        R = Rbat*(1+C*f)
        
        # Calculate resistive losses
        Ploss = (Ibat**2)*R
        
        # Energy loss from power draw
        eloss = np.dot(I,Ploss)
        
        # Cap the battery charging to not be more than the battery can store and adjust after
        # This needs to be replaced by a vectorized operation soon
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